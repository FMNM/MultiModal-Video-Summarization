import os
import json
import whisper
from transformers import pipeline
import yt_dlp as youtube_dl
from bertopic import BERTopic
from tqdm import tqdm
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


# Function to download the audio using yt-dlp
def download_audio(video_link, output_audio_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": ["-ar", "16000"],
        "prefer_ffmpeg": True,
        "keepvideo": False,
        "outtmpl": output_audio_path + ".%(ext)s",
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_link])
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise


# Function to transcribe audio and extract segments using Whisper
def transcribe_audio_with_timestamps(audio_path, model_name="base"):
    try:
        model = whisper.load_model(model_name)
        logger.info("Transcribing audio...")
        result = model.transcribe(audio_path)
        transcription = result["text"]
        segments = result["segments"]  # Contains timestamps per segment
        return transcription, segments
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise


# Summarization using transformer-based summarization (e.g., BART)
def summarize_text_in_chunks(text, chunk_size=1024):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    # Split text into chunks
    text_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    logger.info("Summarizing text in chunks...")
    for chunk in tqdm(text_chunks, desc="Summarizing"):
        try:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}")
            summaries.append("")
    return " ".join(summaries)


# Function to dynamically segment the transcription into topics using BERTopic
def topic_segmentation(segments):
    texts = [segment["text"] for segment in segments]
    topic_model = BERTopic()

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
            }
        topic_segments[topic]["text"].append(segments[idx]["text"])
        topic_segments[topic]["end_time"] = segments[idx]["end"]  # Update the end time

    return topic_segments, topic_model


# Generate summaries for each segmented topic
def summarize_topics(topic_segments, chunk_size=1024):
    topic_summaries = {}
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

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

    return topic_summaries


# Function to create embeddings for segments
def create_embeddings(topic_summaries):
    logger.info("Creating embeddings for topic summaries...")
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


# Main function to process video, transcribe, segment by topic, and summarize
def process_lecture_video(video_link, audio_output_path):
    # Ensure the output directory exists
    output_dir = Path(audio_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download the audio from the video
    logger.info("STEP 1: Downloading audio...")
    download_audio(video_link, audio_output_path)

    # Step 2: Transcribe the audio and get timestamps for each segment
    logger.info("STEP 2: Transcribing audio...")
    transcription, segments = transcribe_audio_with_timestamps(audio_output_path + ".wav")

    # Step 3: Generate overall summary of the lecture
    logger.info("STEP 3: Generating overall summary...")
    overall_summary = summarize_text_in_chunks(transcription, chunk_size=1024)

    # Step 4: Segment the transcription by topics dynamically
    logger.info("STEP 4: Segmenting transcription into topics...")
    topic_segments, topic_model = topic_segmentation(segments)

    # Step 5: Summarize each segmented topic
    logger.info("STEP 5: Summarizing each topic...")
    topic_summaries = summarize_topics(topic_segments)

    return overall_summary, topic_summaries, topic_model
