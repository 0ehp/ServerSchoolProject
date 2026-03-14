import io
import json
import os
from typing import Any
from flask import Flask, request, jsonify
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import essentia.standard as es
import time

print("Starting Server")

app = Flask(__name__)

# ---------- GLOBAL: preload essentia models once at startup ----------
_genre_embed_model = None
_genre_predict_model = None
_genre_labels = None
_rhythm_extractor = es.RhythmExtractor2013()
_key_extractor = es.KeyExtractor()
_loudness_extractor = es.Loudness()
_centroid_extractor = es.SpectralCentroidTime()


MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def get_genre_models():
    global _genre_embed_model, _genre_predict_model, _genre_labels
    if _genre_embed_model is None:
        _genre_embed_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=os.path.join(MODELS_DIR, 'discogs-effnet-bs64-1.pb'),
            output='PartitionedCall:1'
        )
        _genre_predict_model = es.TensorflowPredict2D(
            graphFilename=os.path.join(MODELS_DIR, 'genre_discogs400-discogs-effnet-1.pb'),
            input='serving_default_model_Placeholder',
            output='PartitionedCall:0'
        )
        with open(os.path.join(MODELS_DIR, 'genre_discogs400-discogs-effnet-1.json')) as f:
            _genre_labels = json.load(f)['classes']
    return _genre_embed_model, _genre_predict_model, _genre_labels



def load_audio(wav_bytes, target_sr=16000):
    """decode wav bytes to mono float32 numpy array"""
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # resample if needed
    if sr != target_sr:
        resampler = es.Resample(inputSampleRate=sr, outputSampleRate=target_sr)
        audio = resampler(audio)
    return audio


def load_audio_16k(wav_bytes):
    """load at 16kHz for genre models"""
    return load_audio(wav_bytes, target_sr=16000)


def extract_features(wav_bytes):
    try:
        t0 = time.perf_counter()

        audio = load_audio(wav_bytes)
        t1 = time.perf_counter()

        bpm, beats, beats_confidence, _, beats_intervals = _rhythm_extractor(audio)
        t2 = time.perf_counter()

        key, scale, key_strength = _key_extractor(audio)
        t3 = time.perf_counter()

        loudness = _loudness_extractor(audio)
        t4 = time.perf_counter()

        centroid = _centroid_extractor(audio)
        t5 = time.perf_counter()

        print(
            f"decode={t1-t0:.3f}s "
            f"tempo={t2-t1:.3f}s "
            f"key={t3-t2:.3f}s "
            f"loudness={t4-t3:.3f}s "
            f"centroid={t5-t4:.3f}s"
        )

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

def predict_genre(wav_bytes, top_k=5):
    """predict genre using essentia discogs model"""
    try:
        audio = load_audio_16k(wav_bytes)
        embed_model, predict_model, labels = get_genre_models()

        embeddings = embed_model(audio)
        predictions = predict_model(embeddings)

        mean_preds = predictions.mean(axis=0)
        top_idx = mean_preds.argsort()[-top_k:][::-1]

        return [
            {"label": labels[i], "score": round(float(mean_preds[i]), 4)}
            for i in top_idx
        ]
    except Exception as e:
        print(f"Error predicting genre: {e}")
        return None


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/features")
def single_features():
    try:
        wav_bytes = request.data
    except Exception:
        print("Client disconnected during upload")
        return jsonify({"error": "client disconnected during upload"}), 499

    if not wav_bytes:
        return jsonify({"error": "empty body"}), 400

    result = extract_features(wav_bytes)
    if result is None:
        return jsonify({"error": "feature extraction failed"}), 500

    return jsonify(result), 200

@app.post("/features/batch")
def extract_batch():
    files = request.files.getlist("files")
    wav_bytes_list = [f.read() for f in files]
    print(f"Received {len(files)} files")

    results = [extract_features(w) for w in wav_bytes_list]

    results = [r for r in results if r is not None]
    print(f"Returning {len(results)} results")
    return jsonify(results)


@app.post("/classify")
def classify():
    wav_bytes = request.data
    if not wav_bytes:
        return jsonify({"error": "empty body"}), 400

    result = predict_genre(wav_bytes)
    if result is None:
        return jsonify({"error": "classification failed"}), 500
    return jsonify(result)


@app.post("/classify/batch")
def classify_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    file_data = []
    for f in files:
        wav_bytes = f.read()
        if wav_bytes:
            file_data.append((f.filename, wav_bytes))

    if not file_data:
        return jsonify({"error": "no valid audio files uploaded"}), 400

    def classify_one(item):
        filename, wav_bytes = item
        preds = predict_genre(wav_bytes)
        return {
            "name": filename,
            "preds": preds
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(classify_one, file_data))

    return jsonify(results)

