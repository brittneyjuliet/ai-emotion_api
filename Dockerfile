# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for audio & OpenSMILE
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# (Optional) Preload your Hugging Face model
# ----------------------------
RUN python -c "\
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor; \
Wav2Vec2ForSequenceClassification.from_pretrained('brittneyjuliet/ascended-intelligence-model'); \
Wav2Vec2FeatureExtractor.from_pretrained('brittneyjuliet/ascended-intelligence-model') \
"

# Copy project source code
COPY . .

# Set env vars so Docker knows where HF cache should live
# ENV HF_HOME=/tmp/huggingface_cache
# ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
# ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface_cache

# Expose FastAPI port
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
