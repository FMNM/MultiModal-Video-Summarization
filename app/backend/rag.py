import openai
import os
import json
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from logger import get_logger 
from utils import (
    load_data,
    save_data,
)

logger = get_logger(__name__)

# Load API key from environment variable or config file
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    config_path = Path(__file__).parent / "config.json"
    if config_path.is_file():
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            openai.api_key = config.get("OPENAI_API_KEY")
    else:
        raise ValueError("OpenAI API key not found in environment variables or config.json")

logger.info(f"API_KEY={openai.api_key[:10]}...")

# Function to create embeddings for segments
def create_embeddings(topic_summaries, session_path, force=False):
    logger.info("Creating embeddings for topic summaries...")

    embedding_filepath = Path(session_path) / "embeddings.json"

    already_exist = embedding_filepath.is_file()

    logger.info(f"transcription:{embedding_filepath} segments:{embedding_filepath} force:{force}, already_exist:{already_exist}")

    if already_exist and not force:
        embeddings = load_data(embedding_filepath)
    else:
        embeddings = {}
        for topic, data in tqdm(topic_summaries.items(), desc="Embedding topics"):
            try:
                response = openai.Embedding.create(
                    input=data["full_text"],
                    model="text-embedding-ada-002",
                )
                embeddings[topic] = {
                    "embedding": response["data"][0]["embedding"],
                    "start_time": data["start_time"],
                    "end_time": data["end_time"],
                    "text": data["full_text"],
                    "summary": data["summary"],
                }
                save_data(embeddings, embedding_filepath)
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                embeddings[topic] = None
    return embeddings

# Function to handle queries using RAG
def query_with_rag(question, embeddings):
    logger.info("Processing query with RAG...")
    try:
        # Create embedding for the question
        question_embedding_response = openai.Embedding.create(
            input=question,
            model="text-embedding-ada-002",
        )
        question_embedding = question_embedding_response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error creating question embedding: {e}")
        return None, []

    # Compute similarities
    similarities = []
    for topic, data in embeddings.items():
        if data and data.get("embedding"):
            sim = cosine_similarity(
                [question_embedding],
                [data["embedding"]],
            )[
                0
            ][0]
            similarities.append((sim, topic))

    if not similarities:
        logger.warning("No similarities found.")
        return None, []

    # Get the most relevant topics
    similarities.sort(reverse=True)
    top_topics = [topic for _, topic in similarities[:3]]  # Get top 3 relevant topics

    # Combine the texts of the most relevant topics
    context = " ".join([embeddings[topic]["text"] for topic in top_topics if embeddings[topic]])

    # Use the context to answer the question
    try:
        response = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"Question: {question}"},
            ],
            model="gpt-4o",  # Use 'gpt-4o' or your available model
        )
        answer = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer = "I'm sorry, I couldn't generate an answer at this time."

    # Get relevant timeframes
    relevant_timeframes = [(embeddings[topic]["start_time"], embeddings[topic]["end_time"]) for topic in top_topics if embeddings[topic]]

    return answer, relevant_timeframes
