import streamlit as st
from pathlib import Path
import yt_dlp

# Import backend functions from the corresponding files
from audio import download_audio, transcribe_audio_with_timestamps
from rag_test2 import create_embeddings, query_with_rag
from text_test1 import summarize_text_in_chunks, smart_segment_transcript, summarize_topics, generate_topic_labels_for_clusters
from utils import sha256
from logger import get_logger

# Logger for logging purposes
logger = get_logger(__name__)

# Function to download video using yt-dlp
def download_video(video_url, output_path):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download best video and best audio
        'outtmpl': output_path,  # Specify the output path
        'merge_output_format': 'mp4',  # Ensure that video and audio are merged into an mp4 file
        'quiet': True,  # Suppress yt-dlp output
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Helper function to convert seconds to minutes and seconds
def format_timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{int(seconds):02d}"

# Function to process the lecture video
def process_lecture_video(video_link, session_path, force):

    # Ensure the output directory exists
    output_dir = Path(session_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_file_name = sha256(video_link)  # filename is sha256
    audio_output_path = f"{session_path}/{audio_file_name}"

    # Step 1: Download the audio from the video
    logger.info("STEP 1: Downloading audio...")
    audio_filename = download_audio(video_link, audio_output_path, force=force)

    # Step 2: Transcribe the audio and get timestamps for each segment
    logger.info("STEP 2: Transcribing audio...")
    transcription, segments = transcribe_audio_with_timestamps(audio_filename, session_path, force=force)

    # Step 3: Generate overall summary of the lecture
    logger.info("STEP 3: Generating overall summary...")
    overall_summary = summarize_text_in_chunks(transcription, session_path, force=force, chunk_size=2048)

    # Step 4: Segment the transcription into topics using semantic similarity
    logger.info("STEP 4: Segmenting transcription into topics...")
    topic_segments = smart_segment_transcript(segments, session_path)

    # Step 5: Generate topic labels for each segment using Flan-T5
    logger.info("STEP 5: Generating topic labels...")
    topics = generate_topic_labels_for_clusters(topic_segments, session_path)

    # Step 6: Summarize each segmented topic using BART
    logger.info("STEP 6: Summarizing each topic...")
    topic_summaries = summarize_topics(topics, session_path, force=force)

    return overall_summary, topic_summaries

# Streamlit App

# Set page configuration
st.set_page_config(page_title="Educational Video Summariser", page_icon="🎥", layout="wide")

# Title
st.title("🎥 Educational Video Summariser")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Video Processing", "Summaries", "Question and Answer"])

# Video Processing Tab
with tab1:
    st.header("📥 Video Processing")

    # Input fields
    video_link = st.text_input("Enter YouTube Video Link:", key="video_link")
    session_path = st.text_input("Session Path:", "temp/session", key="session_path")
    force = st.checkbox("Force reprocessing", value=False, key="force")

    # Check if the video path is already in session state
    video_output_path = st.session_state.get("video_output_path", None)

    if st.button("Process Video"):
        if video_link.strip():
            video_output_path = f"{session_path}/downloaded_video.mp4"
            st.session_state["video_output_path"] = video_output_path  # Store video path in session state

            # Check if video is already downloaded or needs to be reprocessed
            with st.spinner("Downloading video... This may take a while."):
                try:
                    download_video(video_link, video_output_path)
                    st.success("Video downloaded successfully.")
                except Exception as e:
                    st.error(f"An error occurred while downloading the video: {e}")
                    logger.error(f"Video download error: {e}")

            # Once video is downloaded, start processing
            with st.spinner("Processing video... This may take a while."):
                try:
                    overall_summary, topic_summaries = process_lecture_video(video_link, session_path, force=force)
                    embeddings = create_embeddings(topic_summaries, session_path, force=force)
                    st.success("Processing completed.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    logger.error(f"Processing error: {e}")
                    st.stop()  # Stop execution if an error occurs

            # Store results in session state
            st.session_state["overall_summary"] = overall_summary
            st.session_state["topic_summaries"] = topic_summaries
            st.session_state["embeddings"] = embeddings

    # Embed the processed video in the Streamlit app (show only if it exists)
    if video_output_path and Path(video_output_path).exists():
        st.video(video_output_path)

# Summaries Tab
with tab2:
    st.header("📝 Summaries")

    if "overall_summary" in st.session_state:
        # First display the topic summaries
        st.subheader("Topic Overview")
        topic_summaries = st.session_state["topic_summaries"]
        video_file = st.session_state.get("video_output_path")  # Path to the downloaded video

        for topic, data in topic_summaries.items():
            # Convert start and end times to mm:ss format
            start_time_formatted = format_timestamp(data['start_time'])
            end_time_formatted = format_timestamp(data['end_time'])

            # Generate seekable links for start and end times
            start_time_seconds = int(data['start_time'])  # Start time in seconds for seeking
            end_time_seconds = int(data['end_time'])  # End time in seconds for seeking
            start_time_link = f"{video_file}#t={start_time_seconds}"
            end_time_link = f"{video_file}#t={end_time_seconds}"

            with st.expander(f"{topic}"):
                st.markdown(f"**Summary:**<br>{data['summary']}", unsafe_allow_html=True)
                st.write(f"**Start Time:** [**{start_time_formatted}**]({start_time_link})")
                st.write(f"**End Time:** [**{end_time_formatted}**]({end_time_link})")
                st.markdown(f"**Full Text:**<br>{data['full_text']}", unsafe_allow_html=True)


            # Render LaTeX if the topic contains any math expressions
            if "math_expression" in data:  # Assume 'math_expression' is a key for LaTeX content
                st.latex(data["math_expression"])

        # Then display the overall summary with better formatting
        st.subheader("Overall Summary")
        formatted_summary = f"""
        <div style='padding: 10px; border-radius: 8px;'>
            <h4 >📝 Overview:</h4>
            <p>{st.session_state["overall_summary"]}</p>
        </div>
        """
        st.markdown(formatted_summary, unsafe_allow_html=True)

    else:
        st.info("Please process a video first in the 'Video Processing' tab.")

# Streamlit App - Question and Answer Tab
with tab3:
    st.header("❓ Question and Answer")

    if "embeddings" in st.session_state:
        question = st.text_input("Enter your question about the lecture:", key="question")

        if st.button("Get Answer", key="get_answer"):
            if question.strip() == "":
                st.error("Please enter a valid question.")
            else:
                with st.spinner("Retrieving answer..."):
                    # Unpack answer and relevant_info (which contains topic, label, start_time, end_time)
                    answer, relevant_info = query_with_rag(question, st.session_state["embeddings"])

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Relevant Topics and Timeframes")
                for idx, info in enumerate(relevant_info):
                    # Convert start and end times to mm:ss format
                    start_time_formatted = format_timestamp(info['start_time'])
                    end_time_formatted = format_timestamp(info['end_time'])

                    # Generate video link for both start and end times
                    video_file = st.session_state.get("video_output_path")  # Path to your video file
                    start_time_seconds = int(info['start_time'])  # Start time in seconds for seeking
                    end_time_seconds = int(info['end_time'])  # End time in seconds for seeking
                    start_time_link = f"{video_file}#t={start_time_seconds}"
                    end_time_link = f"{video_file}#t={end_time_seconds}"

                    # Display topic with clickable start and end times
                    st.markdown(
                        f"**{info['topic']}**: [**{start_time_formatted}**]({start_time_link}) - [**{end_time_formatted}**]({end_time_link})"
                    )

    else:
        st.info("Please process a video first in the 'Video Processing' tab.")
