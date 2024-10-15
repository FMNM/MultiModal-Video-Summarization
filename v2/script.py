import os
import openai
import torch
import whisper
from transformers import pipeline
import yt_dlp as youtube_dl
from keybert import KeyBERT
from bertopic import BERTopic
from tqdm import tqdm  # For progress bars


# Function to download the audio using yt-dlp
def download_audio(video_link, output_audio_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_audio_path,  # Save the file with the specified path
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_link])


# Function to transcribe audio and extract segments using Whisper
def transcribe_audio_with_timestamps(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    transcription = result["text"]
    segments = result["segments"]  # Contains timestamps per segment
    return transcription, segments


# Summarization using transformer-based summarization (e.g., BART)
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]["summary_text"]


# Set up OpenAI API Key for ChatGPT
openai.api_key = "your_openai_api_key_here"


# Function to ask questions using RAG-like structure with ChatGPT API
def query_chatgpt(context, question):
    response = openai.Completion.create(engine="gpt-4", prompt=f"{context}\n\nQ: {question}\nA:", max_tokens=150, temperature=0.5)
    return response.choices[0].text.strip()


# Function to dynamically segment the transcription into topics using BERTopic
def topic_segmentation(transcription, segments):
    texts = [segment["text"] for segment in segments]
    topic_model = BERTopic()  # Initialize BERTopic model

    print("Segmenting topics...")
    # Use tqdm to show progress while segmenting the topics
    topics, _ = topic_model.fit_transform(tqdm(texts, desc="Topic segmentation"))

    # Group segments by topics with timeframes
    topic_segments = {}
    for idx, topic in enumerate(topics):
        if topic not in topic_segments:
            topic_segments[topic] = {"text": [], "start_time": segments[idx]["start"], "end_time": segments[idx]["end"]}
        topic_segments[topic]["text"].append(segments[idx]["text"])
        topic_segments[topic]["end_time"] = segments[idx]["end"]  # Update the end time

    return topic_segments, topic_model


# Generate summaries for each segmented topic
def summarize_topics(topic_segments):
    topic_summaries = {}
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

    print("Summarizing topics...")
    for topic, data in tqdm(topic_segments.items(), desc="Summarizing each topic"):
        full_text = " ".join(data["text"])  # Combine all texts for this topic
        summary = summarizer(full_text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
        topic_summaries[topic] = {"summary": summary, "start_time": data["start_time"], "end_time": data["end_time"]}

    return topic_summaries


# Main function to process video, transcribe, segment by topic, summarize, and enable Q&A with ChatGPT
def process_lecture_video(video_link, audio_output_path):
    # Step 1: Download the audio from the video
    download_audio(video_link, audio_output_path)

    # Step 2: Transcribe the audio and get timestamps for each segment
    transcription, segments = transcribe_audio_with_timestamps(audio_output_path)

    # Step 3: Generate overall summary of the lecture
    overall_summary = summarize_text(transcription)

    # Step 4: Segment the transcription by topics dynamically
    topic_segments, topic_model = topic_segmentation(transcription, segments)

    # Step 5: Summarize each segmented topic
    topic_summaries = summarize_topics(topic_segments)

    return overall_summary, topic_summaries, topic_model


# Function to handle queries, providing both answer and relevant timeframe
def query_with_timeframe(question, transcript, segments, topic_summaries):
    # Answer the question using ChatGPT API
    answer = query_chatgpt(transcript, question)

    # Find relevant topic timeframe from answer
    relevant_timeframe = None
    for topic, summary in topic_summaries.items():
        if answer.lower() in summary["summary"].lower():
            relevant_timeframe = (summary["start_time"], summary["end_time"])
            break

    return answer, relevant_timeframe


# Example Usage:
video_link = "your_youtube_or_custom_video_link_here"
audio_output_path = "downloaded_audio.mp3"

# Process the lecture video to get summaries and topics
overall_summary, topic_summaries, topic_model = process_lecture_video(video_link, audio_output_path)

# Output the overall summary and topic summaries
print("Overall Summary of the lecture:", overall_summary)
for topic, summary in topic_summaries.items():
    print(f"Topic {topic}: {summary['summary']} (Time: {summary['start_time']} to {summary['end_time']})")

# Example: Ask a query and get answer with timeframe
question = "What is the main topic discussed?"
answer, timeframe = query_with_timeframe(question, overall_summary, topic_summaries)
print(f"Answer: {answer}, Relevant Timeframe: {timeframe}")
