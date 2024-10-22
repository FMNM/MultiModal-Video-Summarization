import streamlit as st
from pathlib import Path

# Import backend functions from the corresponding files
from audio import download_audio, transcribe_audio_with_timestamps
from rag_test1 import create_embeddings, query_with_rag  # Ensure rag_test1.py is correctly updated
from text_test import summarize_text_in_chunks, topic_segmentation, summarize_topics
from utils import sha256
from logger import get_logger

# Logger for logging purposes
logger = get_logger(__name__)

# Function to process the lecture video
def process_lecture_video(video_link, session_path, force):
    output_dir = Path(session_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_file_name = sha256(video_link)  # filename is sha256 of video link
    audio_output_path = f"{session_path}/{audio_file_name}"
    
    # Step 1: Download audio
    logger.info("STEP 1: Downloading audio...")
    audio_filename = download_audio(video_link, audio_output_path, force=force)
    
    # Step 2: Transcribe audio
    logger.info("STEP 2: Transcribing audio...")
    transcription, segments = transcribe_audio_with_timestamps(audio_filename, session_path, force=force)
    
    # Step 3: Generate overall summary
    logger.info("STEP 3: Generating overall summary...")
    overall_summary = summarize_text_in_chunks(transcription, session_path, force=force, chunk_size=2048)
    
    # Step 4: Segment by topics
    logger.info("STEP 4: Segmenting transcription into topics...")
    topic_segments, topic_model = topic_segmentation(segments, session_path, force=True)  # Set force=True here
    
    # Step 5: Summarize topics (Set force=True to regenerate summaries)
    logger.info("STEP 5: Summarizing each topic...")
    topic_summaries = summarize_topics(topic_segments, session_path, force=True)  # Set force=True here

    # Ensure topic_ids are strings
    topic_summaries = {str(topic_id): data for topic_id, data in topic_summaries.items()}

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
    video_link = st.text_input("Enter YouTube Video Link:", key='video_link')
    session_path = st.text_input("Session Path:", "temp/session", key='session_path')
    force = st.checkbox("Force reprocessing", value=False, key='force')

    # Process Video Button
    if st.button("Process Video", key='process_video'):
        if video_link.strip() == "":
            st.error("Please enter a valid YouTube video link.")
        else:
            with st.spinner("Processing video... This may take a while."):
                try:
                    # Process the video and create embeddings
                    overall_summary, topic_summaries = process_lecture_video(
                        video_link, session_path, force=force)
                    embeddings = create_embeddings(topic_summaries, session_path, force=force)
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    logger.error(f"Processing error: {e}")
                    st.session_state.clear()
                    st.stop()  # Use st.stop() instead of return

            st.success("Processing completed.")

            # Store results in session state
            st.session_state['overall_summary'] = overall_summary
            st.session_state['topic_summaries'] = topic_summaries
            st.session_state['embeddings'] = embeddings

# Summaries Tab
with tab2:
    st.header("📝 Summaries")

    if 'overall_summary' in st.session_state:
        st.subheader("Overall Summary")
        st.write(st.session_state['overall_summary'])

        st.subheader("Topic Summaries")
        topic_summaries = st.session_state['topic_summaries']
        for topic_id_str, data in topic_summaries.items():
            # Use get method to handle missing 'topic_label'
            topic_label = data.get('topic_label', f"Topic {topic_id_str}")
            with st.expander(f"{topic_label}"):
                st.write(f"**Summary:** {data.get('summary', 'No summary available.')}")
                st.write(f"**Start Time:** {data.get('start_time', 'N/A')}")
                st.write(f"**End Time:** {data.get('end_time', 'N/A')}")
                st.write(f"**Full Text:** {data.get('full_text', 'No text available.')}")
    else:
        st.info("Please process a video first in the 'Video Processing' tab.")

# Question and Answer Tab
with tab3:
    st.header("❓ Question and Answer")

    if 'embeddings' in st.session_state:
        question = st.text_input("Enter your question about the lecture:", key='question')

        if st.button("Get Answer", key='get_answer'):
            if question.strip() == "":
                st.error("Please enter a valid question.")
            else:
                with st.spinner("Retrieving answer..."):
                    try:
                        answer, timeframes = query_with_rag(question, st.session_state['embeddings'])
                    except Exception as e:
                        st.error(f"An error occurred while retrieving the answer: {e}")
                        logger.error(f"RAG retrieval error: {e}")
                        st.stop()

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Relevant Timeframes")
                topic_summaries = st.session_state['topic_summaries']

                for timeframe in timeframes:
                    # Each timeframe is a tuple: (topic_id_str, start_time, end_time)
                    topic_id_str, start_time, end_time = timeframe
                    # Ensure topic_id_str is a string
                    topic_info = topic_summaries.get(topic_id_str, {})
                    topic_label = topic_info.get('topic_label', f"Topic {topic_id_str}")
                    st.write(f"**{topic_label}:** {start_time} - {end_time}")
    else:
        st.info("Please process a video first in the 'Video Processing' tab.")
