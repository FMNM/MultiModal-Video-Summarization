import streamlit as st
from pathlib import Path

# Import backend functions from the corresponding files
from audio import download_audio, transcribe_audio_with_timestamps
from rag_test import create_embeddings, query_with_rag
from text import summarize_text_in_chunks, smart_segment_transcript, summarize_topics, generate_topic_labels_for_clusters
from utils import sha256
from logger import get_logger

# Logger for logging purposes
logger = get_logger(__name__)


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

    # Process Video Button
    if st.button("Process Video", key="process_video"):
        if video_link.strip() == "":
            st.error("Please enter a valid YouTube video link.")
        else:
            with st.spinner("Processing video... This may take a while."):
                try:
                    # Process the video and create embeddings
                    overall_summary, topic_summaries, topic_model = process_lecture_video(video_link, session_path, force=force)
                    embeddings = create_embeddings(topic_summaries, session_path, force=force)
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    logger.error(f"Processing error: {e}")
                    st.session_state.clear()
                    st.stop()  # Use st.stop() instead of return

            st.success("Processing completed.")

            # Store results in session state
            st.session_state["overall_summary"] = overall_summary
            st.session_state["topic_summaries"] = topic_summaries
            st.session_state["embeddings"] = embeddings

# Summaries Tab
with tab2:
    st.header("📝 Summaries")

    if "overall_summary" in st.session_state:
        st.subheader("Overall Summary")
        st.write(st.session_state["overall_summary"])

        st.subheader("Topic Summaries")
        topic_summaries = st.session_state["topic_summaries"]
        for topic, data in topic_summaries.items():
            with st.expander(f"{topic}"):
                st.write(f"**Summary:** {data['summary']}")
                st.write(f"**Start Time:** {data['start_time']}")
                st.write(f"**End Time:** {data['end_time']}")
                st.write(f"**Full Text:** {data['full_text']}")
    else:
        st.info("Please process a video first in the 'Video Processing' tab.")

# Question and Answer Tab
with tab3:
    st.header("❓ Question and Answer")

    if "embeddings" in st.session_state:
        question = st.text_input("Enter your question about the lecture:", key="question")

        if st.button("Get Answer", key="get_answer"):
            if question.strip() == "":
                st.error("Please enter a valid question.")
            else:
                with st.spinner("Retrieving answer..."):
                    answer, timeframes = query_with_rag(question, st.session_state["embeddings"])

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Relevant Timeframes")
                for idx, (start_time, end_time) in enumerate(timeframes):
                    st.write(f"Topic {idx + 1}: {start_time} - {end_time}")
    else:
        st.info("Please process a video first in the 'Video Processing' tab.")
