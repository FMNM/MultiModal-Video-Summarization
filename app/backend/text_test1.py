from transformers import pipeline
from tqdm import tqdm
import logging
import os
import re
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import json
from openai import OpenAI
from openai import OpenAIError


from utils import time_check_decorator, save_data, load_data, sha256
from gpu_options import device, device_index
from logger import get_logger

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

# Function to estimate the number of tokens based on text length
def estimate_tokens(text):
    # Assuming roughly 4 characters per token, including spaces and punctuation
    return len(text) // 4

# Function to format the summary using OpenAI API
def format_summary(summary_text):
    # Estimate the number of tokens in the input summary
    input_tokens = estimate_tokens(summary_text)

    # Adjust max_tokens for the output, ensuring it's within limits (max 4096 for GPT-4)
    # You can tweak this based on your desired output length and model's capabilities
    max_output_tokens = max(1500, 4096 - input_tokens)  # Ensure a minimum of 1500 output tokens

    prompt = f"""
    Format the following summary to improve readability and make it look beautiful:
    - maintain the original summary size while expanding on key concepts from the summary
    - remove this line from the summary:For more travel photography, visit CNN.com/Travel for weekly Travel Snapshots galleries, or explore trending travel photography at the Daily Mail
    - don't give the summary a title.

    Summary:
    {summary_text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the engine you prefer, e.g., "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that formats summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_output_tokens,  # Dynamic max tokens
            temperature=0.5,  # Controls creativity, lower is more focused
            n=1,
            stop=None
        )

        # Extract formatted summary
        formatted_summary = response.choices[0].message.content.strip()
        return formatted_summary

    except OpenAIError as e:
        logger.error(f"Error while formatting summary using OpenAI API: {e}")
        return summary_text  # Return the original summary if there's an error


# Summarization using transformer-based summarization (e.g., BART)
@time_check_decorator
def summarize_text_in_chunks(text, session_path, force=True, chunk_size=2048, min_length=80, max_length=300):
    overall_summary_filepath = Path(f"{session_path}/overall_summary.txt")

    already_exist = overall_summary_filepath.is_file()

    logger.info(f"overall_summary_filepath:{overall_summary_filepath} force:{force}, already_exist:{already_exist}")

    if already_exist and not force:
        logger.info(f"{overall_summary_filepath} already exists, loading existing data.")
        overall_summary = load_data(overall_summary_filepath)
    else:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_index)

        sentences = re.split(r"(?<=[.!?]) +", text)
        text_chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                text_chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            text_chunks.append(current_chunk.strip())

        save_data("\n".join(text_chunks), Path(f"{session_path}/text_chunks.txt"))

        logger.info(f"len:len(text_chunks)")

        for text_chunk in text_chunks:
            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.debug(f"len:{len(text_chunk)} chunk:{text_chunk}")
            else:
                logger.info(f"len:{len(text_chunk)} chunk:{text_chunk[:100]}...")

        summaries = []

        logger.info("Summarizing text in chunks...")
        for chunk in tqdm(text_chunks, desc="Summarizing"):
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            summaries.append(summary)

        # Combine the summaries into an overall summary
        overall_summary = "\n".join(summaries)

        # Log the combined summary before OpenAI API formatting
        logger.debug(f"Combined summary before OpenAI API formatting: {overall_summary[:500]}")

        # Call OpenAI API to format the overall summary
        overall_summary = format_summary(overall_summary)
        
        # Log the formatted summary
        logger.debug(f"Formatted overall summary: {overall_summary[:500]}")

        # Save the raw overall summary
        save_data(overall_summary, overall_summary_filepath)

    return overall_summary


def smart_segment_transcript(segments, session_path, similarity_threshold=0.65, min_cluster_size=60, max_cluster_size=180):
    """
    Perform context-aware clustering of transcript segments based on sentence similarity.
    """

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    clusters = []
    current_cluster = [segments[0]]  # Initialize the first cluster
    current_cluster_embeddings = [embedding_model.encode(segments[0]["text"], convert_to_tensor=True)]

    for i in range(1, len(segments)):
        segment_embedding = embedding_model.encode(segments[i]["text"], convert_to_tensor=True)

        # Calculate average embedding of the current cluster
        cluster_embedding = torch.mean(torch.stack(current_cluster_embeddings), dim=0)

        # Compute cosine similarity between the current segment and the cluster
        similarity = util.cos_sim(cluster_embedding, segment_embedding).item()

        if similarity >= similarity_threshold and len(current_cluster) < max_cluster_size:
            # Add to the current cluster if similar enough and within size limits
            current_cluster.append(segments[i])
            current_cluster_embeddings.append(segment_embedding)
        else:
            # Save the completed cluster if it meets minimum size criteria
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)

                # Start a new cluster with the current segment
                current_cluster = [segments[i]]
                current_cluster_embeddings = [segment_embedding]
            else:
                # If the cluster is too small, continue adding the segment to the same cluster
                current_cluster.append(segments[i])
                current_cluster_embeddings.append(segment_embedding)

    # Add the final cluster
    if current_cluster:
        clusters.append(current_cluster)

    # Save the clusters to a JSON file
    save_data(clusters, Path(f"{session_path}/topic_segments.json"))

    return clusters


# Function to get text from topic_segments and generate encapsulating topics
def generate_topic_labels_for_clusters(clusters, session_path):
    """Generate topic labels using Flan-T5 for each segment cluster."""

    # Load models for topic extraction and sentence embeddings
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    flan_t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    topics = []
    logger.info("Generating topic labels for clusters...")

    for cluster in tqdm(clusters, desc="Processing clusters"):
        combined_text = " ".join([segment["text"] for segment in cluster])

        # Generate a concise topic label using Flan-T5
        inputs = tokenizer.encode(
            "Generate a topic for: " + combined_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        try:
            summary_ids = flan_t5_model.generate(
                inputs,
                max_length=12,
                min_length=3,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            topic_label = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating topic label: {e}")
            topic_label = "Unknown Topic"

        # Extract start and end time from the cluster
        start_time = cluster[0]["start"]
        end_time = cluster[-1]["end"]

        topic_info = {
            "topic_label": topic_label,
            "start_time": start_time,
            "end_time": end_time,
            "texts": [segment["text"] for segment in cluster],
        }
        topics.append(topic_info)

    # Save the topics to a JSON file
    save_data(topics, Path(f"{session_path}/topics.json"))

    return topics


# Generate summaries for each segmented topic with OpenAI API formatting
def summarize_topics(topics, session_path, force=True, chunk_size=1024):
    """
    Generate summaries for each topic using BART and format using OpenAI API.
    """
    topic_summaries_filepath = Path(f"{session_path}/topic_summaries.json")
    already_exist = topic_summaries_filepath.is_file()

    logger.info(f"topic_summaries_filepath: {topic_summaries_filepath} force: {force}, already_exist: {already_exist}")

    if already_exist and not force:
        logger.info(f"{topic_summaries_filepath} already exists, loading existing data.")
        return load_data(topic_summaries_filepath)
    elif already_exist and force:
        logger.info("Force reprocessing set to true, regenerating summaries.")

    topic_summaries = {}
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_index)

    logger.info("Summarizing topics...")

    # Ensure topics is a list
    if not isinstance(topics, list):
        raise ValueError("Expected 'topics' to be a list of dictionaries.")

    for topic_info in tqdm(topics, desc="Summarizing each topic"):
        # Ensure topic_info is a dictionary
        if not isinstance(topic_info, dict):
            raise ValueError(f"Expected a dictionary for topic_info, but got {type(topic_info)}")

        if "texts" not in topic_info:
            raise KeyError(f"Missing 'texts' key in topic_info: {topic_info}")

        # Combine all texts for the topic into a single string
        full_text = " ".join(topic_info["texts"])

        # Split the full text into manageable chunks
        text_chunks = [full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)]

        summaries = []
        for chunk in text_chunks:
            try:
                # Generate a summary for each chunk using BART
                summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing topic chunk: {e}")
                summaries.append("")

        # Combine all summaries into one
        combined_summary = " ".join(summaries)

        # Log the combined summary before OpenAI API formatting
        logger.debug(f"Combined summary before OpenAI API call: {combined_summary[:500]}")

        # Use OpenAI API to format the combined summary
        formatted_summary = format_summary(combined_summary)

        # Log the formatted summary after OpenAI API
        logger.debug(f"Formatted summary from OpenAI API: {formatted_summary[:500]}")

        # Store the summary with relevant metadata
        topic_label = topic_info.get("topic_label", f"Topic {len(topic_summaries) + 1}")

        topic_summaries[topic_label] = {
            "summary": formatted_summary,  # Use the formatted summary from OpenAI
            "start_time": topic_info["start_time"],
            "end_time": topic_info["end_time"],
            "full_text": full_text,  # Store original text for RAG retrieval
        }

    # Save the summaries to a JSON file
    save_data(topic_summaries, topic_summaries_filepath)

    return topic_summaries


