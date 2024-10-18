# Re-export the function
__all__ = ["process_lecture_video", "create_embedding", "query_with_rag"] 

from rag import (
    create_embeddings,
    query_with_rag 
)

from text import (
    summarize_text_in_chunks,
    topic_segmentation,
    summarize_topics
)

from audio import (
    download_audio, 
    transcribe_audio_with_timestamps
)

from logger import get_logger 
from pathlib import Path
from utils import sha256

logger = get_logger(__name__)

# Main function to process video, transcribe, segment by topic, and summarize
def process_lecture_video(video_link, session_path, force):

    # Ensure the output directory exists
    output_dir = Path(session_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_file_name = sha256(video_link) # filename is sha256
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

    # Step 4: Segment the transcription by topics dynamically
    logger.info("STEP 4: Segmenting transcription into topics...")
    topic_segments, topic_model = topic_segmentation(segments, session_path, force=force)

    # Step 5: Summarize each segmented topic
    logger.info("STEP 5: Summarizing each topic...")
    topic_summaries = summarize_topics(topic_segments, session_path, force=force)

    return overall_summary, topic_summaries, topic_model


if __name__ == "__main__":

    sample_video_link = "https://www.youtube.com/watch?v=AhyznRSDjw8"
    session_path = "temp/this_is_sample_session_path"

    force=False

    overall_summary, topic_summaries, _ = process_lecture_video(sample_video_link, session_path, force=force)

    embeddings = create_embeddings(topic_summaries, session_path, force=force)

    question = "What is the main topic of the lecture?"

    # Get the answer and relevant timeframes using RAG
    answer, timeframes = query_with_rag(question, embeddings)

    print("\n"*5)
    print(f"overall_summary:{overall_summary}")
    print("\n"*5)
    print(f"question:{question}")
    print(f"answer:{answer}")
    print(f"timeframes:{timeframes}")