# app.py

# working code before wav conversion

# import tempfile
# import numpy as np
# import torchaudio
# import torch
# import opensmile
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
# from huggingface_hub import hf_hub_download
# from safetensors.torch import load_file as load_safetensors  # Safetensors loader

# # ----------------------------
# # Initialize FastAPI app
# # ----------------------------
# app = FastAPI(title="Emotion Detection API")

# # Enable CORS so browser frontends (like p5.js) can call this API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080"],  # For dev; restrict to your site in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------
# # Frequency Analyzer
# # ----------------------------
# class FrequencyAnalyzer:
#     def __init__(self):
#         self.smile = opensmile.Smile(
#             feature_set=opensmile.FeatureSet.emobase,
#             feature_level=opensmile.FeatureLevel.Functionals,
#         )

#     def extract_frequency(self, audio_chunk: np.ndarray, sr: int) -> float:
#         """Extracts the average fundamental frequency (F0) from an audio signal."""
#         try:
#             signal = audio_chunk.reshape(1, -1) if audio_chunk.ndim == 1 else audio_chunk
#             features = self.smile.process_signal(signal=signal, sampling_rate=sr)
#             for col in features.columns:
#                 if "F0" in col and "amean" in col:
#                     val = float(features[col].iloc[0])
#                     if np.isfinite(val) and val > 0:
#                         return val
#         except Exception:
#             pass
#         return float("nan")


# # ----------------------------
# # Emotion Analyzer (Wav2Vec2)
# # ----------------------------
# class Wav2Vec2EmotionAnalyzer:
#     def __init__(self, model_repo="brittneyjuliet/ascended-intelligence-model"):
#         print(f"🔄 Loading model from Hugging Face Hub: {model_repo}")
#         self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_repo)
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_repo)

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#         self.id2label = {
#             0: "anger_fear",
#             1: "joy_excited",
#             2: "sadness",
#             3: "curious_reflective",
#             4: "calm_content",
#         }

#     def analyze_emotion(self, audio_path: str) -> str:
#         """Classifies emotion from a .wav audio file."""
#         try:
#             speech_array, sr = torchaudio.load(audio_path)
#             if speech_array.shape[0] > 1:
#                 speech_array = torch.mean(speech_array, dim=0, keepdim=True)
#             if sr != 16000:
#                 resampler = torchaudio.transforms.Resample(sr, 16000)
#                 speech_array = resampler(speech_array)
#             speech = speech_array.squeeze().numpy()

#             inputs = self.feature_extractor(
#                 speech, sampling_rate=16000, return_tensors="pt", padding=True
#             )
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             with torch.no_grad():
#                 logits = self.model(**inputs).logits
#             predicted_id = torch.argmax(logits, dim=-1).item()
#             return self.id2label[predicted_id]
#         except Exception as e:
#             print(f"Emotion analysis error: {e}")
#             return "Unknown"


# # ----------------------------
# # Initialize analyzers
# # ----------------------------
# freq_analyzer = FrequencyAnalyzer()
# emotion_analyzer = Wav2Vec2EmotionAnalyzer()


# # ----------------------------
# # Routes
# # ----------------------------
# @app.get("/")
# def home():
#     return {"message": "Emotion detection API is running."}


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     """
#     Accepts an uploaded .wav audio file and returns detected emotion + frequency.
#     """
#     try:
#         # Save uploaded file to a temp path
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(await file.read())
#             tmp_path = tmp.name

#         # Load audio
#         waveform, sr = torchaudio.load(tmp_path)
#         signal = waveform.squeeze().numpy()

#         # Analyze frequency + emotion
#         freq = freq_analyzer.extract_frequency(signal, sr)
#         emotion = emotion_analyzer.analyze_emotion(tmp_path)

#         return {
#             "emotion": emotion,
#             "frequency_hz": None if np.isnan(freq) else round(freq, 2),
#         }

#     except Exception as e:
#         return {"error": str(e)}

import tempfile
import os
import numpy as np
import torchaudio
import torch
import opensmile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment

# ---------------------------------------------------------
# 🔧 1. Fix cache permissions for Hugging Face Spaces
# ---------------------------------------------------------
# CACHE_DIR = "/tmp/huggingface_cache"
# os.environ["HF_HOME"] = CACHE_DIR
# os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
# os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

# os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(title="Emotion Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "https://brittneyjuliet.github.io",
        "https://mzpalmer-cpu.github.io"
    ],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Frequency Analyzer
# ----------------------------
class FrequencyAnalyzer:
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_frequency(self, audio_chunk: np.ndarray, sr: int) -> float:
        try:
            signal = audio_chunk.reshape(1, -1) if audio_chunk.ndim == 1 else audio_chunk
            features = self.smile.process_signal(signal=signal, sampling_rate=sr)
            for col in features.columns:
                if "F0" in col and "amean" in col:
                    val = float(features[col].iloc[0])
                    if np.isfinite(val) and val > 0:
                        return val
        except Exception:
            pass
        return float("nan")


# ----------------------------
# Emotion Analyzer
# ----------------------------
class Wav2Vec2EmotionAnalyzer:
    def __init__(self, model_repo="brittneyjuliet/ascended-intelligence-model"):
        print(f"🔄 Loading model from Hugging Face Hub: {model_repo}")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_repo)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_repo)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.id2label = {
            0: "ANGER_FEAR",
            1: "JOY_EXCITED",
            2: "SADNESS",
            3: "CURIOUS_REFLECTIVE",
            4: "CALM_CONTENT",
        }

    def analyze_emotion(self, audio_path: str) -> str:
        try:
            speech_array, sr = torchaudio.load(audio_path)
            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                speech_array = resampler(speech_array)
            speech = speech_array.squeeze().numpy()

            inputs = self.feature_extractor(
                speech, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            return self.id2label[predicted_id]
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return "Unknown"


# ----------------------------
# Initialize analyzers
# ----------------------------
freq_analyzer = FrequencyAnalyzer()
emotion_analyzer = Wav2Vec2EmotionAnalyzer()


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return {"message": "Emotion detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a browser-recorded .wav file and returns emotion + frequency.
    Converts WAV to PCM16 mono 16kHz before processing.
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Convert WAV to standard PCM16 mono 16kHz
        converted_path = tmp_path.replace(".wav", "_converted.wav")
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(converted_path, format="wav")

        # Load audio
        waveform, sr = torchaudio.load(converted_path)
        signal = waveform.squeeze().numpy()

        # Analyze frequency + emotion
        freq = freq_analyzer.extract_frequency(signal, sr)
        emotion = emotion_analyzer.analyze_emotion(converted_path)

        # Clean up
        os.remove(tmp_path)
        os.remove(converted_path)

        return {
            "emotion": emotion,
            "frequency_hz": None if np.isnan(freq) else round(freq, 2),
        }

    except Exception as e:
        return {"error": str(e)}


