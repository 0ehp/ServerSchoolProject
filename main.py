import io
import json
import os
from fastapi import FastAPI, Request
import numpy as np
import soundfile as sf
import essentia.standard as es
from pydantic import BaseModel
import time

print("Starting Server")

app = FastAPI()

DEBUG_PRINTING = False
TARGET_SR = 16000
MAX_SECONDS = 30
MAX_SAMPLES = TARGET_SR * MAX_SECONDS

_genre_embed_model = None
_genre_predict_model = None
_genre_labels = None
_rhythm_extractor = es.RhythmExtractor2013(method="degara")
_key_extractor = es.KeyExtractor()
_loudness_extractor = es.Loudness()
_centroid_extractor = es.SpectralCentroidTime()
_windowing = es.Windowing(type='hann')
_danceability_extractor = es.Danceability()
_dynamic_complexity_extractor = es.DynamicComplexity()
_mfcc_algo = es.MFCC(numberCoefficients=13)
_pitch_algo = es.PitchYinFFT(frameSize=2048, sampleRate=44100)
_spectrum = es.Spectrum()
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

class SongResult(BaseModel):
    bpm: float
    beats_confidence: float
    key: str
    scale: str
    key_strength: float
    loudness: float
    spectral_centroid: float
    danceability: float
    dynamic_complexity: float
    mfcc: list[float]
    vocal: float


def extract_features(wav_bytes):
    try:
        t0 = time.perf_counter()
        audio = load_audio(wav_bytes)
        t1 = time.perf_counter()

        bpm, beats, beats_confidence, _, beats_intervals = _rhythm_extractor(audio)
        t2 = time.perf_counter()

        key, scale, key_strength = _key_extractor(audio)
        loudness = _loudness_extractor(audio)
        centroid = _centroid_extractor(audio)
        t3 = time.perf_counter()

        dance, dfa_array = _danceability_extractor(audio)
        t4 = time.perf_counter()

        complexity, loudness_dc = _dynamic_complexity_extractor(audio)
        t5 = time.perf_counter()

        mfcc_frames = []
        confidence = 0
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            windowed = _windowing(frame)
            spec = _spectrum(windowed)
            bands, mfcc_coeffs = _mfcc_algo(spec)
            mfcc_frames.append(mfcc_coeffs)
            pitch, confidence = _pitch_algo(spec)

        mfcc_mean = np.mean(mfcc_frames, axis=0).tolist()
        t6 = time.perf_counter()

        print(
            f"load: {t1 - t0:.3f}s | bpm: {t2 - t1:.3f}s | key+loud+centroid: {t3 - t2:.3f}s | dance: {t4 - t3:.3f}s | complexity: {t5 - t4:.3f}s | mfcc: {t6 - t5:.3f}s | total: {t6 - t0:.3f}s")
        return SongResult(
            bpm=float(bpm),
            beats_confidence=float(beats_confidence),
            key=key,
            scale=scale,
            key_strength=float(key_strength),
            loudness=float(loudness),
            spectral_centroid=float(centroid),
            danceability=float(dance),
            dynamic_complexity=float(complexity),
            mfcc=mfcc_mean,
            vocal=float(confidence)
        )

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def predict_genre(wav_bytes, top_k=5):
    """predict genre using essentia discogs model"""
    try:
        audio = load_audio(wav_bytes)
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
async def single_features(request: Request):
    try:
        wav_bytes = await request.body()
    except Exception:
        print("Client disconnected during upload")
        return {"error": "client disconnected during upload"}

    if not wav_bytes:
        return {"error": "empty body"}

    result = extract_features(wav_bytes)
    if result is None:
        return {"error": "feature extraction failed"}

    return result

@app.post("/features/batch")
async def extract_batch(request: Request):
    wav_bytes_list = await request.body()

    results = [extract_features(w) for w in wav_bytes_list]

    results = [r for r in results if r is not None]

    if DEBUG_PRINTING:
        print(f"Returning {len(results)} results")
    return results


@app.post("/classify")
async def classify(request: Request):
    wav_bytes = await request.body()
    if not wav_bytes:
        return {"error": "empty body"}

    result = predict_genre(wav_bytes)
    if result is None:
        return {"error": "classification failed"}
    return result


