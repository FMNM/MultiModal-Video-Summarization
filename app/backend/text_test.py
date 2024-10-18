import os
import re
import logging
import json
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from bertopic import BERTopic
#import openai
from openai import OpenAI

from utils import (
    time_check_decorator,
    save_data,
    load_data,
    sha256
)

from gpu_options import (
    device,
    device_index
)

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

# Summarization using transformer-based summarization (e.g., BART)
@time_check_decorator
def summarize_text_in_chunks(text, session_path, force=True, chunk_size=2048, min_length=80, max_length=300):
    overall_summary_filepath = Path(f"{session_path}/overall_summary.txt")

    already_exist = overall_summary_filepath.is_file()

    logger.info(f"overall_summary_filepath:{overall_summary_filepath} force:{force}, already_exist:{already_exist}")

    if already_exist and not force:
        logger.info(f"{overall_summary_filepath} already exists.")
        overall_summary = load_data(overall_summary_filepath)
    else:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_index)

        sentences = re.split(r'(?<=[.!?]) +', text)
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

        logger.info(f"Number of text chunks: {len(text_chunks)}")

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

        overall_summary = "\n".join(summaries)

        save_data(overall_summary, overall_summary_filepath)

    return overall_summary

# Function to dynamically segment the transcription into topics using BERTopic and generate topic titles with OpenAI API
def topic_segmentation(segments, session_path, force=True):
    topic_segments_filepath = Path(f"{session_path}/topic_segments.json")

    texts = [segment["text"] for segment in segments]

    already_exist = topic_segments_filepath.is_file()

    logger.info(f"topic_segments_filepath:{topic_segments_filepath} force:{force}, already_exist:{already_exist}")

    if already_exist and not force:
        logger.info(f"{topic_segments_filepath} already exists.")
        topic_segments = load_data(topic_segments_filepath)
        topic_model = None  # You can load the topic_model if you saved it
    else:
        logger.info("Segmenting topics...")
        topic_model = BERTopic()
        try:
            topics, _ = topic_model.fit_transform(texts)
        except Exception as e:
            logger.error(f"Error during topic segmentation: {e}")
            raise

        # Group segments by topics with timeframes
        topic_segments = {}
        for idx, topic in enumerate(topics):
            if topic not in topic_segments:
                topic_segments[topic] = {
                    "text": [],
                    "start_time": segments[idx]["start"],
                    "end_time": segments[idx]["end"],
                    "topic_info": topic_model.get_topic(topic),
                    "segment_indices": [],
                }
            topic_segments[topic]["text"].append(segments[idx]["text"])
            topic_segments[topic]["end_time"] = segments[idx]["end"]  # Update the end time
            topic_segments[topic]["segment_indices"].append(idx)

        # Generate topic labels using OpenAI API
        logger.info("Generating topic labels using OpenAI API...")
        for topic_id, data in topic_segments.items():
            # Use the top n words from topic_info to create a prompt
            topic_keywords = [word for word, _ in data["topic_info"][:5]]  # Top 5 keywords
            keywords_text = ", ".join(topic_keywords)
            prompt = (
                f"Based on the following keywords, provide a concise and descriptive title for the topic "
                f"without any quotation marks. The title should be less than 10 words:\nKeywords: {keywords_text}"
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates topic titles."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=20,  # Increased from 10 to 20
                    n=1,
                    temperature=0.5,
                )
                # Remove leading and trailing quotation marks
                topic_label = response.choices[0].message.content.strip().strip('"').strip("'")
                topic_segments[topic_id]["topic_label"] = topic_label
            except Exception as e:
                logger.error(f"Error generating topic label: {e}")
                topic_segments[topic_id]["topic_label"] = f"Topic {topic_id}"

        save_data(topic_segments, topic_segments_filepath)

    return topic_segments, topic_model

# Generate summaries for each segmented topic
def summarize_topics(topic_segments, session_path, force=True, chunk_size=1024):
    topic_summaries_filepath = Path(f"{session_path}/topic_summaries.json")

    already_exist = topic_summaries_filepath.is_file()

    logger.info(f"topic_summaries_filepath:{topic_summaries_filepath} force:{force}, already_exist:{already_exist}")

    if already_exist and not force:
        logger.info(f"{topic_summaries_filepath} already exists.")
        topic_summaries = load_data(topic_summaries_filepath)
    else:
        topic_summaries = {}
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_index)

        logger.info("Summarizing topics...")
        for topic_id, data in tqdm(topic_segments.items(), desc="Summarizing each topic"):
            full_text = " ".join(data["text"])

            # Split the full text into chunks to handle large inputs
            text_chunks = [full_text[i: i + chunk_size] for i in range(0, len(full_text), chunk_size)]
            summaries = []

            for chunk in text_chunks:
                try:
                    summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error summarizing topic chunk: {e}")
                    summaries.append("")

            # Combine chunk summaries and store them
            topic_summaries[topic_id] = {
                "topic_label": data.get("topic_label", f"Topic {topic_id}"),
                "summary": " ".join(summaries),
                "start_time": data["start_time"],
                "end_time": data["end_time"],
                "full_text": full_text,  # Added for RAG retrieval
            }

        save_data(topic_summaries, topic_summaries_filepath)

    return topic_summaries
