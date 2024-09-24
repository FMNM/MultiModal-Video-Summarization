# Modified the blip implementation
import os
import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from transformers import VivitConfig, VivitModel, BlipForConditionalGeneration, BlipProcessor, pipeline
import whisper
import yt_dlp as youtube_dl
from ultralytics import YOLO #import YOLOv5  # Ensure YOLOv5 is installed
from PIL import Image

# Load the dataset (replace with your actual dataset path)
dataset = pd.read_csv('dataset/youtube.csv')

# Function to download video using yt-dlp
def download_video(video_link, output_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_link])

# Step 1: Download the video
video_link = dataset['link'][299]
video_path = 'downloaded_video.mp4'
output_folder = 'video_frames'
download_video(f"https://www.youtube.com/watch?v={video_link}", video_path)

# Step 2: Extract frames from the video
def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        count += 1
    
    cap.release()

extract_frames(video_path, output_folder, frame_rate=30)

# Step 3: Preprocess frames for ViViT (Video Vision Transformer) model
def preprocess_frame(frame, size=(224, 224)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

def preprocess_all_frames(output_folder, size=(224, 224)):
    preprocessed_frames = []
    for frame_file in sorted(os.listdir(output_folder)):
        frame_path = os.path.join(output_folder, frame_file)
        frame = cv2.imread(frame_path)
        preprocessed_frame = preprocess_frame(frame, size)
        preprocessed_frames.append(preprocessed_frame)
    preprocessed_frames = np.vstack(preprocessed_frames)
    return preprocessed_frames

preprocessed_frames = preprocess_all_frames(output_folder)

# Function to transcribe audio from video using Whisper
def transcribe_audio(video_path):
    model = whisper.load_model("medium")  # You can use different sizes: "tiny", "base", "small", "medium", "large"
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


# Step 4: Advanced Feature Extraction using ViViT
def extract_vivit_features(preprocessed_frames, num_frames_per_segment=3):
    torch.cuda.empty_cache()
    configuration = VivitConfig(image_size=224, num_frames=num_frames_per_segment)
    model = VivitModel(configuration)
    model.eval()

    # Convert frames to tensor
    frames_tensor = torch.tensor(preprocessed_frames, dtype=torch.float32)
    batch_size = frames_tensor.shape[0] // num_frames_per_segment
    frames_tensor = frames_tensor[:batch_size * num_frames_per_segment].reshape(batch_size, num_frames_per_segment, 3, 224, 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_tensor = frames_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(frames_tensor)

    video_features = outputs.last_hidden_state  # (batch_size, num_frames_per_segment, hidden_dim)
    return video_features

# Step 5: Scene Transition Extraction with Advanced Contextual Feature Changes
def compute_feature_differences(features):
    """
    Compute the differences between consecutive feature vectors across time.
    Args:
        features: tensor of shape (num_frames, feature_dim)
    Returns:
        diff: a tensor representing the difference between consecutive frame features
    """
    # Flatten the frames (combine temporal and spatial dimensions)
    num_frames = features.shape[0]
    feature_dim = features.shape[-1]
    
    # Reshape to (num_frames, feature_dim) 
    features = features.view(num_frames, -1).cpu().numpy()

    # Compute pairwise feature distance (Euclidean or Cosine) between consecutive frames
    feature_diffs = np.linalg.norm(np.diff(features, axis=0), axis=1)

    return feature_diffs

def detect_scene_transitions(features, threshold=0.7):
    """
    Detect significant scene transitions based on feature differences.
    Args:
        features: tensor of shape (num_frames, feature_dim)
        threshold: threshold value for detecting scene change
    Returns:
        scene_transition_indices: list of indices where scene transitions occur
    """
    feature_diffs = compute_feature_differences(features)

    # Plot differences for visualization
    plt.plot(feature_diffs)
    plt.title('Feature Differences Between Consecutive Frames')
    plt.xlabel('Frame')
    plt.ylabel('Feature Difference')
    plt.show()

    # Scene transitions are where the feature difference exceeds the threshold
    scene_transition_indices = np.where(feature_diffs > threshold)[0]
    
    return scene_transition_indices

# Step 6: Apply Temporal Coherence via Agglomerative Clustering
def cluster_scene_transitions(features, scene_transition_indices, num_clusters=5):
    """
    Cluster the detected scene transitions into coherent groups using Agglomerative Clustering.
    Args:
        features: tensor of shape (num_frames, feature_dim)
        scene_transition_indices: indices of detected scene transitions
        num_clusters: number of scene clusters
    Returns:
        clustered_scenes: list of indices corresponding to clustered scene transitions
    """
    # Extract the transition features
    transition_features = features[scene_transition_indices].cpu().numpy()

    # Use Agglomerative Clustering with temporal constraint (Ward linkage)
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(transition_features)

    clustered_scenes = []
    for cluster_label in range(num_clusters):
        cluster_indices = np.where(clustering_model.labels_ == cluster_label)[0]
        clustered_scenes.append(scene_transition_indices[cluster_indices[0]])  # Select the first frame in the cluster

    return clustered_scenes

# Step 7: Object Detection with YOLOv5
yolo_model = YOLO("yolov5xu.pt", device="cuda" if torch.cuda.is_available() else "cpu")

def detect_objects_in_frame(frame):
    # YOLOv5 expects PIL images
    pil_frame = Image.fromarray(frame)
    results = yolo_model(pil_frame)
    detected_objects = [res['name'] for res in results.xyxy[0]]
    return detected_objects

# Step 8: Generate descriptions with YOLOv5 and BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

descriptions = []
for i, transition_frame in enumerate(clustered_scenes):
    key_frame = preprocessed_frames[transition_frame]
    key_frame_img_uint8 = (key_frame * 255).astype(np.uint8)
    
    # Detect objects using YOLOv5
    detected_objects = detect_objects_in_frame(key_frame_img_uint8)
    
    # BLIP Description Generation
    inputs = blip_processor(images=key_frame_img_uint8, return_tensors="pt")
    
    # Modify the generation parameters to increase descriptiveness
    outputs = blip_model.generate(**inputs, max_new_tokens=250, num_beams=10, length_penalty=2.0, no_repeat_ngram_size=2)
    
    visual_description = blip_processor.decode(outputs[0], skip_special_tokens=True)
    
    # Combine detected objects and BLIP description
    combined_description = f"Objects detected: {', '.join(detected_objects)}. Visual: {visual_description}"
    
    descriptions.append(f"Scene {i+1}: {combined_description}")
    print(f"Description for Scene {i+1}: {combined_description}")


# Step 9: Combine Descriptions and Audio Transcription
video_transcript = transcribe_audio(video_path)  # Transcribe the video
combined_text = f"{video_transcript} {' '.join(descriptions)}"

# Step 10: Final Summary using summarization model
summarizer = pipeline("summarization", model="facebook/mbart-large-50", device=-1)  # Use mBART for summarization
summary_text = summarize_text(combined_text, summarizer)

print("Generated Summary:")
print(summary_text)

