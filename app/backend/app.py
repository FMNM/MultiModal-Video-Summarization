from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import platform
from methods import (
    process_lecture_video,
    create_embeddings,
    query_with_rag,
)
from uuid import uuid4

# Flask application
app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

# In-memory data store (dictionary) for storing user data
user_data_store = {}


# Endpoint to process video and generate summaries
@app.route("/process", methods=["POST"])
def process_video():
    data = request.get_json()
    video_link = data.get("video_link")
    if not video_link:
        return jsonify({"error": "No video link provided."}), 400

    force = True if data.get("force").lower() == "true" else False

    # Generate a unique session ID
    session_id = str(uuid4())
    session["session_id"] = session_id

    session_path = f"temp/{session_id}"

    # Process video and generate topic summaries
    try:
        overall_summary, topic_summaries, _ = process_lecture_video(video_link, session_path, force)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Create embeddings for RAG
    embeddings = create_embeddings(topic_summaries)

    # Store results in the user data store
    user_data_store[session_id] = {
        "overall_summary": overall_summary,
        "topic_summaries": topic_summaries,
        "embeddings": embeddings,
    }

    # Return summaries and session ID
    return jsonify({"overall_summary": overall_summary, "topic_summaries": topic_summaries, "session_id": session_id}), 200


# Endpoint to handle query and return answers with relevant timeframes
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")
    session_id = data.get("session_id")

    if not question or not session_id:
        return jsonify({"error": "Question and session ID are required."}), 400

    # Retrieve embeddings from the user data store
    user_data = user_data_store.get(session_id)
    if not user_data:
        return jsonify({"error": "Session not found."}), 404

    embeddings = user_data.get("embeddings")
    if not embeddings:
        return jsonify({"error": "Embeddings not found for this session."}), 404

    # Get the answer and relevant timeframes using RAG
    answer, timeframes = query_with_rag(question, embeddings)

    return jsonify({"answer": answer, "timeframes": timeframes}), 200


if __name__ == "__main__":
    # Use 5001 on macOS as 5000 is occupied by Control Center
    port = 5001 if platform.system() == "Darwin" else 5000

    # Ensure the temp directory exists
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=port)
