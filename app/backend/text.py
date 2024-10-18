from transformers import pipeline
from bertopic import BERTopic
from tqdm import tqdm
import logging
import os
import re
from pathlib import Path

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

# Summarization using transformer-based summarization (e.g., BART)

@time_check_decorator
def summarize_text_in_chunks(text, session_path, force=True, chunk_size=2048, min_length=80, max_length=300):
    overall_summary_filepath = Path(f"{session_path}/overall_summary.txt")

    already_exist = overall_summary_filepath.is_file()

    logger.info(f"overall_summary_filepath:{overall_summary_filepath} force:{force}, already_exist:{already_exist}")

    if already_exist:
        logger.info(f"{overall_summary_filepath} already exists.")

    if not force and already_exist:
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

        save_data("\n".join(text_chunks), f"{session_path}/text_chunks.txt")
        
        logger.info(f"len:len(text_chunks)")

        for text_chunk in text_chunks :
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

# Function to dynamically segment the transcription into topics using BERTopic
def topic_segmentation(segments, session_path, force=True):
    topic_segments_filepath = Path(f"{session_path}/topic_segments.json")

    texts = [segment["text"] for segment in segments]

    already_exist = topic_segments_filepath.is_file()

    logger.info(f"topic_segments_filepath:{topic_segments_filepath} force:{force}, already_exist:{already_exist}")

    topic_model = BERTopic()

    if already_exist:
        logger.info("{topic_segments_filepath} already exist}")
    
    if not force and already_exist:
        topic_segments = load_data(topic_segments_filepath)
    else:
        logger.info("Segmenting topics...")
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
                    "topic_info": topic_model.get_topic(topic)
                }
            topic_segments[topic]["text"].append(segments[idx]["text"])
            topic_segments[topic]["end_time"] = segments[idx]["end"]  # Update the end time

        save_data(topic_segments, topic_segments_filepath)

    return topic_segments, topic_model


# Generate summaries for each segmented topic
def summarize_topics(topic_segments, session_path, force=True, chunk_size=1024):
    topic_summaries_filepath = Path(f"{session_path}/topic_summaries.json")

    already_exist = topic_summaries_filepath.is_file()

    logger.info(f"topic_summaries_filepath:{topic_summaries_filepath} force:{force}, already_exist:{already_exist}")

    if already_exist:
        logger.info(f"{topic_summaries_filepath} already exists.")
    
    if not force and already_exist:
        topic_summaries = load_data(topic_summaries_filepath)
    else:
        topic_summaries = {}
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_index)

        logger.info("Summarizing topics...")
        for topic, data in tqdm(topic_segments.items(), desc="Summarizing each topic"):
            full_text = " ".join(data["text"])

            # Split the full text into chunks to handle large inputs
            text_chunks = [full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)]
            summaries = []

            for chunk in text_chunks:
                try:
                    summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error summarizing topic chunk: {e}")
                    summaries.append("")

            # Combine chunk summaries and store them
            topic_summaries[topic] = {
                "summary": " ".join(summaries),
                "start_time": data["start_time"],
                "end_time": data["end_time"],
                "full_text": full_text,  # Added for RAG retrieval
            }

        save_data(topic_summaries, topic_summaries_filepath)

    return topic_summaries
