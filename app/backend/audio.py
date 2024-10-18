import yt_dlp as youtube_dl
import logging
import os
import whisper
from pathlib import Path

from utils import (
    time_check_decorator,
    save_data,
    load_data,
    sha256
)


logger = logging.getLogger(__name__)

# Function to download the audio using yt-dlp
@time_check_decorator
def download_audio(video_link, output_audio_path, force):

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
        "outtmpl": output_audio_path + ".%(ext)s"

    }

    # Be caful that opts.preferredcodec is "wav"
    audio_filename = output_audio_path + ".wav" 

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:

            already_exist = os.path.exists(audio_filename) 

            logger.info(f"audio_filename:{audio_filename}, force:{force}, already_exist:{already_exist}")

            if already_exist:
                logger.info(f"The audio file already exists")
            if force or not already_exist:
                logger.info(f"Downloading the video then change it into audio: {audio_filename}")
                ydl.download([video_link])
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise

    return audio_filename

# Function to transcribe audio and extract segments using Whisper
@time_check_decorator
def transcribe_audio_with_timestamps(audio_path, session_path, force = False,model_name="base"):
    try:

        transcription_filepath = Path(f"{session_path}/transcription.txt")
        segments_filepath = Path(f"{session_path}/segments.json")


        already_exist = transcription_filepath.is_file() and segments_filepath.is_file()

        logger.info(f"transcription:{transcription_filepath} segments:{segments_filepath} force:{force}, already_exist:{already_exist}")

        if not force and already_exist:
            print("Loading data from file")
            transcription = load_data(transcription_filepath)
            segments = load_data(segments_filepath)
        else:
            model = whisper.load_model(model_name)
            logger.info("Transcribing audio...")
            result = model.transcribe(audio_path)
            transcription = result["text"]
            segments = result["segments"]  # Contains timestamps per segment

            save_data(transcription, transcription_filepath)
            save_data(segments, segments_filepath)
        
        return transcription, segments
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise
