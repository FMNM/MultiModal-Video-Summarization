import os
import cv2
import torch.cuda
import yt_dlp as youtube_dl
import numpy as np
import pandas as pd
import torch
from transformers import VivitConfig, VivitModel
from sklearn.cluster import KMeans
import whisper
from transformers import BlipForConditionalGeneration, BlipProcessor, pipeline

# Load the dataset
dataset = pd.read_csv('dataset/youtube.csv')

# Function to download video using yt-dlp
def download_video(video_link, output_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_link])

# Function to extract frames from a video
def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at the specified rate
        if count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        count += 1
    
    cap.release()

# Function to preprocess frames
def preprocess_frame(frame, size=(224, 224)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to preprocess all frames
def preprocess_all_frames(output_folder, size=(224, 224)):
    preprocessed_frames = []
    for frame_file in sorted(os.listdir(output_folder)):
        frame_path = os.path.join(output_folder, frame_file)
        frame = cv2.imread(frame_path)
        preprocessed_frame = preprocess_frame(frame, size)
        preprocessed_frames.append(preprocessed_frame)
    preprocessed_frames = np.vstack(preprocessed_frames)
    return preprocessed_frames

# Function to transcribe audio from video using Whisper
def transcribe_audio(video_path):
    model = whisper.load_model("base")  # You can use different sizes: "tiny", "base", "small", "medium", "large"
    result = model.transcribe(video_path)
    transcription = result['text']
    return transcription

# Function to summarize text in smaller chunks with truncation
def summarize_text(text, summarizer, chunk_size=256, max_length=1024):  # Reduced chunk size to 256
    tokens = text.split()
    summaries = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i:i + chunk_size])
        if len(chunk) > max_length:
            chunk = chunk[:max_length]  # Truncate if it exceeds max_length
        summary = summarizer(chunk, max_length=max_length, min_length=int(chunk_size / 2), do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

# Example usage
video_link = dataset['link'][299]
video_path = 'downloaded_video.mp4'
output_folder = 'video_frames'

# Step 1: Download the video
download_video(f"https://www.youtube.com/watch?v={video_link}", video_path)

# Step 2: Extract frames from the video
extract_frames(video_path, output_folder, frame_rate=30)

# Step 3: Preprocess all frames
preprocessed_frames = preprocess_all_frames(output_folder)
print("Preprocessed frames shape:", preprocessed_frames.shape)

# Step 4: Feature Extraction with ViViT
torch.cuda.empty_cache()
configuration = VivitConfig(image_size=224, num_frames=3)
model = VivitModel(configuration)

model.eval()

frames_tensor = torch.tensor(preprocessed_frames, dtype=torch.float32)
num_frames_per_segment = 3
batch_size = frames_tensor.shape[0] // num_frames_per_segment
frames_tensor = frames_tensor[:batch_size * num_frames_per_segment].reshape(batch_size, num_frames_per_segment, 3, 224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frames_tensor = frames_tensor.to(device)
model = model.to(device)

with torch.no_grad():
    outputs = model(frames_tensor)

video_features = outputs.last_hidden_state
print("Video features shape:", video_features.shape)

# Step 5: Key Frame Selection
num_frames, feature_dim = video_features.shape[1], video_features.shape[-1]
video_features_reshaped = video_features.view(num_frames, -1).cpu().numpy()

k = 5
kmeans = KMeans(n_clusters=k, random_state=0).fit(video_features_reshaped)

key_frame_indices = []
for i in range(k):
    cluster_center = kmeans.cluster_centers_[i]
    distances = np.linalg.norm(video_features_reshaped - cluster_center, axis=1)
    key_frame_idx = np.argmin(distances)
    key_frame_indices.append(key_frame_idx)

key_frames = [preprocessed_frames[idx] for idx in key_frame_indices]

for i, key_frame in enumerate(key_frames):
    if key_frame.shape[0] == 1:
        key_frame_img = key_frame.squeeze(0)
    else:
        key_frame_img = key_frame

    if key_frame_img.shape[0] == 3:
        key_frame_img = np.transpose(key_frame_img, (1, 2, 0))
    
    cv2.imwrite(f'key_frame_{i}.jpg', key_frame_img * 255)

# Step 6: Generate Descriptions for Key Frames using BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

descriptions = []
for i, key_frame in enumerate(key_frames):
    key_frame_img_uint8 = (key_frame * 255).astype(np.uint8)
    inputs = blip_processor(images=key_frame_img_uint8, return_tensors="pt")
    outputs = blip_model.generate(**inputs, max_new_tokens=150, num_beams=7)
    description = blip_processor.decode(outputs[0], skip_special_tokens=True)
    descriptions.append(f"Key Frame {i+1}: {description}")
    print(f"Description for Key Frame {i+1}: {description}")

# Step 7: Combine Key Frame Descriptions with Video Transcript
video_transcript = transcribe_audio(video_path)
combined_text = f"{video_transcript} {' '.join(descriptions)}"

# Step 8: Summarize Combined Content
summarizer = pipeline("summarization", model="facebook/mbart-large-50", device=-1)  # Force CPU usage
summary_text = summarize_text(combined_text, summarizer)

print("Generated Summary:")
print(summary_text)
