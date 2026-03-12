import io
import os
import tempfile
from typing import Any
from flask import Flask, request, jsonify
import numpy as np  # number crunching
import soundfile as sf
from laion_clap import CLAP_Module
from waitress import serve
from concurrent.futures import ProcessPoolExecutor
# ---------- GLOBAL: genre classifier pipeline ----------
# Uses dima806/music_genres_classification (wav2vec2-based genre model)

from transformers import pipeline  # HuggingFace pipeline for genre classification

print("Starting Server")

app = Flask(__name__)
@app.get("/")
def health():
    return {"status": "ok"}

MODEL = None

_genre_classifier = None

def get_clap_model():
    global MODEL
    if MODEL is None:
        from laion_clap import CLAP_Module
        MODEL = CLAP_Module(enable_fusion=False)
        MODEL.load_ckpt()
    return MODEL

def get_genre_classifier():
    # Create a pipeline called audio-classification using the model speecifies, device -1 means only use cpu then store in genre_classifier

    global _genre_classifier
    if _genre_classifier is None:
        _genre_classifier = pipeline(
            "audio-classification",
            model="dima806/music_genres_classification",
            device=-1,
        )
    return _genre_classifier


@app.post("/features")
def SingleFeatures() -> dict[str, Any]:

    bio = io.BytesIO(request.data)
    audio, sr = sf.read(bio)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Convert to tensor batch
    import torch
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    model = get_clap_model()

    embedding = model.get_audio_embedding_from_data(
        x=audio_tensor,
        use_tensor=True
    )

    embedding = embedding.detach().cpu().numpy()[0]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return {"Embedding": embedding.tolist()}

def Features(wav_bytes):

    audio, sr = sf.read(wav_bytes)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Convert to tensor batch
    import torch
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    model = get_clap_model()

    embedding = model.get_audio_embedding_from_data(
        x=audio_tensor,
        use_tensor=True
    )

    embedding = embedding.detach().cpu().numpy()[0]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return {"Embedding": embedding.tolist()}


@app.post("/features/batch")
def extract_batch():
	files = request.files.getlist("files")
	wav_bytes_list = [f.read for f in files]
	
	with ProcessPoolExecutor(max_workers=4) as executor:
		results = list(executor.map(Features, wav_bytes_list))
	return jsonify(results)

@app.post("/classify")
def get_music_tags_from_bytes():

    wav_bytes = request.data
    if not wav_bytes:
        return jsonify({"error": "empty body"}), 400

    classifier = get_genre_classifier()
    # Makes a temp file
    audio, sr = sf.read(io.BytesIO(wav_bytes))

    # HF pipeline can take a file path
    res = classifier({"array": audio, "sampling_rate": sr}, top_k=3)

    # res is a list of dicts: { 'label': 'rock', 'score': 0.92, ... }
    return jsonify(res)


@app.post("/classify_batch")
def classifyBatch():

    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    classifier = get_genre_classifier()
    results = []

    for f in files:
        wav_bytes = f.read()
        if not wav_bytes:
            continue

        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(wav_bytes)

            preds = classifier(path, top_k=3)
            results.append({
                "name": f.filename,
                "preds": preds  # list of { label, score }
            })
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    return jsonify(results)


if __name__ == "__main__":
    print("Server is UP")
    # print(app.url_map)
    serve(app, host="0.0.0.0", port=4000, threads=4)
    print("Server is up") 
   # app.run(host="0.0.0.0", port=4000, debug=False) #TODO set to false later
