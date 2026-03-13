import io
import json
import os
from typing import Any
from flask import Flask, request, jsonify
import numpy as np
import soundfile as sf
from waitress import serve
from concurrent.futures import ThreadPoolExecutor
import essentia.standard as es

print("Starting Server")

app = Flask(__name__)

# ---------- GLOBAL: preload essentia models once at startup ----------
_genre_embed_model = None
_genre_predict_model = None
_genre_labels = None

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


def load_audio(wav_bytes, target_sr=44100):
    """decode wav bytes to mono float32 numpy array"""
    audio, sr = sf.read(io.BytesIO(wav_bytes))
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
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
    """extract tempo, key, loudness from audio"""
    try:
        audio = load_audio(wav_bytes)

        # tempo / beat
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        # key
        key_extractor = es.KeyExtractor()
        key, scale, key_strength = key_extractor(audio)

        # loudness
        loudness = es.Loudness()(audio)

        # spectral centroid
        centroid = es.SpectralCentroidTime()(audio)

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
def single_features() -> dict[str, Any]:
    wav_bytes = request.data
    if not wav_bytes:
        return jsonify({"error": "empty body"}), 400

    result = extract_features(wav_bytes)
    if result is None:
        return jsonify({"error": "feature extraction failed"}), 500
    return jsonify(result)


@app.post("/features/batch")
def extract_batch():
    files = request.files.getlist("files")
    wav_bytes_list = [f.read() for f in files]
    print(f"Received {len(files)} files")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(extract_features, wav_bytes_list))

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


@app.post("/classify_batch")
def classify_batch():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    results = []
    for f in files:
        wav_bytes = f.read()
        if not wav_bytes:
            continue
        preds = predict_genre(wav_bytes)
        results.append({
            "name": f.filename,
            "preds": preds
        })
    return jsonify(results)


if __name__ == "__main__":
    print("Server is UP")
    serve(app, host="0.0.0.0", port=4000, threads=4)