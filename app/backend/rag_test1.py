import os
import json
import time
from pathlib import Path

#import openai
from openai import OpenAI
from openai import RateLimitError, OpenAIError
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from logger import get_logger
from utils import load_data, save_data

logger = get_logger(__name__)

# Function to get the API key
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        config_path = Path(__file__).parent / "config.json"
        if config_path.is_file():
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
                api_key = config.get("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key not found in environment variables or config.json")
    return api_key

# Instantiate the OpenAI client
client = OpenAI(
    api_key=get_api_key(),
)

logger.info(f"API_KEY={client.api_key[:10]}...")  # Logs only the first 10 characters for security

# Function to create embeddings for segments
def create_embeddings(topic_summaries, session_path, force=False):
    logger.info("Creating embeddings for topic summaries...")

    embedding_filepath = Path(session_path) / "embeddings.json"

    already_exist = embedding_filepath.is_file()

    logger.info(f"Embedding file: {embedding_filepath}, force: {force}, already_exist: {already_exist}")

    if already_exist and not force:
        embeddings = load_data(embedding_filepath)
        logger.info("Embeddings loaded from existing file.")
    else:
        embeddings = {}
        for topic_id, data in tqdm(topic_summaries.items(), desc="Embedding topics"):
            topic_id_str = str(topic_id)  # Ensure topic_id is a string
            retry_attempts = 3
            for attempt in range(1, retry_attempts + 1):
                try:
                    response = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=data["full_text"],
                    )
                    # Access the embedding from the response
                    #embedding = response['data'][0]['embedding']
                    embedding = response.data[0].embedding  # Alternative
                    embeddings[topic_id_str] = {
                        "embedding": embedding,
                        "start_time": data["start_time"],
                        "end_time": data["end_time"],
                        "text": data["full_text"],
                        "summary": data["summary"],
                    }
                    logger.info(f"Successfully created embedding for topic: {topic_id_str}")
                    break  # Exit retry loop on success
                except RateLimitError as e:
                    logger.warning(f"Rate limit error on attempt {attempt}/{retry_attempts} for topic '{topic_id_str}': {e}")
                    if attempt < retry_attempts:
                        sleep_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying after {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Max retry attempts reached for topic '{topic_id_str}'. Skipping embedding.")
                        embeddings[topic_id_str] = None
                except OpenAIError as e:
                    logger.error(f"OpenAI API error for topic '{topic_id_str}': {e}")
                    embeddings[topic_id_str] = None
                    break  # Do not retry on non-rate limit errors
                except Exception as e:
                    logger.error(f"Unexpected error for topic '{topic_id_str}': {e}")
                    embeddings[topic_id_str] = None
                    break  # Do not retry on unexpected errors

        # Save all embeddings after processing all topics
        save_data(embeddings, embedding_filepath)
        logger.info(f"All embeddings saved to {embedding_filepath}.")

    return embeddings

# Function to handle queries using RAG
def query_with_rag(question, embeddings):
    logger.info("Processing query with RAG...")
    try:
        # Create embedding for the question
        question_embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=question,
        )
        # Access the embedding from the response
        #question_embedding = question_embedding_response['data'][0]['embedding']
        question_embedding = question_embedding_response.data[0].embedding
        logger.info("Successfully created question embedding.")
    except RateLimitError as e:
        logger.error(f"Rate limit error while creating question embedding: {e}")
        return "I'm sorry, the system is currently overloaded. Please try again later.", []
    except OpenAIError as e:
        logger.error(f"OpenAI API error while creating question embedding: {e}")
        return "I'm sorry, I couldn't process your question at this time.", []
    except Exception as e:
        logger.error(f"Unexpected error while creating question embedding: {e}")
        return "I'm sorry, an unexpected error occurred.", []

    # Compute similarities
    similarities = []
    for topic_id_str, data in embeddings.items():
        if data and data.get("embedding"):
            sim = cosine_similarity(
                [question_embedding],
                [data["embedding"]],
            )[0][0]
            similarities.append((sim, topic_id_str))

    if not similarities:
        logger.warning("No similarities found.")
        return "I'm sorry, I couldn't find relevant information to answer your question.", []

    # Get the most relevant topics
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_topics = [topic_id_str for _, topic_id_str in similarities[:3]]  # Get top 3 relevant topics
    logger.info(f"Top topics for the question: {top_topics}")

    # Combine the texts of the most relevant topics
    context = " ".join([embeddings[topic_id_str]["text"] for topic_id_str in top_topics if embeddings[topic_id_str]])
    logger.debug(f"Combined context for the question: {context}")

    # Use the context to answer the question
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Corrected model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"Question: {question}"},
            ],
        )
        # Access the answer from the response
        #answer = response['choices'][0]['message']['content'].strip()
        answer = response.choices[0].message.content.strip()
        logger.info("Successfully generated answer.")
    except RateLimitError as e:
        logger.error(f"Rate limit error while generating answer: {e}")
        answer = "I'm sorry, the system is currently overloaded. Please try again later."
    except OpenAIError as e:
        logger.error(f"OpenAI API error while generating answer: {e}")
        answer = "I'm sorry, I couldn't generate an answer at this time."
    except Exception as e:
        logger.error(f"Unexpected error while generating answer: {e}")
        answer = "I'm sorry, an unexpected error occurred."

    # Get relevant timeframes with topic IDs
    relevant_timeframes = [
        (topic_id_str, embeddings[topic_id_str]["start_time"], embeddings[topic_id_str]["end_time"])
        for topic_id_str in top_topics if embeddings.get(topic_id_str)
    ]
    logger.info(f"Relevant timeframes: {relevant_timeframes}")

    return answer, relevant_timeframes
