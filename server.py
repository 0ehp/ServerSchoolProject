import io
import json
import os
from flask import Flask, request, jsonify
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import essentia.standard as es

print("Starting Server")

app = Flask(__name__)

DEBUG_PRINTING = False
TARGET_SR = 16000
MAX_SECONDS = 30
MAX_SAMPLES = TARGET_SR * MAX_SECONDS

_genre_embed_model = None
_genre_predict_model = None
_genre_labels = None
_rhythm_extractor = es.RhythmExtractor2013()
_key_extractor = es.KeyExtractor()
_loudness_extractor = es.Loudness()
_centroid_extractor = es.SpectralCentroidTime()
_resamplers = {}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def get_genre_models():
    global _genre_embed_model, _genre_predict_model, _genre_labels
    if _genre_embed_model is None:
        _genre_embed_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1"
        )
        _genre_predict_model = es.TensorflowPredict2D(
            graphFilename=os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"
        )
        with open(os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.json")) as f:
            _genre_labels = json.load(f)["classes"]
    return _genre_embed_model, _genre_predict_model, _genre_labels


def get_resampler(input_sr, output_sr):
    key = (input_sr, output_sr)
    if key not in _resamplers:
        _resamplers[key] = es.Resample(
            inputSampleRate=input_sr,
            outputSampleRate=output_sr
        )
    return _resamplers[key]


def load_audio(wav_bytes, target_sr=TARGET_SR):
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = get_resampler(sr, target_sr)(audio)

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    return audio


def load_audio_16k(wav_bytes):
    return load_audio(wav_bytes, target_sr=16000)


def extract_features(wav_bytes):
    try:
        audio = load_audio(wav_bytes)

        bpm, beats, beats_confidence, _, beats_intervals = _rhythm_extractor(audio)
        key, scale, key_strength = _key_extractor(audio)
        loudness = _loudness_extractor(audio)
        centroid = _centroid_extractor(audio)

        return {
            "bpm": round(float(bpm), 2),
            "beats_confidence": round(float(beats_confidence), 2),
            "key": key,
            "scale": scale,
            "key_strength": round(float(key_strength), 2),
            "loudness": round(float(loudness), 2),
            "spectral_centroid": round(float(centroid), 2),
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None