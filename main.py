
import os
import io
import joblib
import librosa
import numpy as np

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "audio_detector_svm.pkl"
STATIC_DIR = BASE_DIR / "static"

# Load trained model once when the server starts
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Audio Deepfake Detector")

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# Same feature extraction used during training
def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)

    features.extend(np.mean(delta, axis=1))
    features.extend(np.std(delta, axis=1))

    # Spectral features
    spectral_features = [
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y)
    ]

    for feature in spectral_features:
        features.append(np.mean(feature))
        features.append(np.std(feature))

    return np.array(features, dtype=np.float32)



@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = {
        ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".mp4"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds 50 MB."
        )

    temp_path = BASE_DIR / f"temp_audio{extension}"

    try:
        # Save uploaded audio temporarily
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Load audio using librosa
        features = extract_features(str(temp_path))
        features = features.reshape(1, -1)

        if not np.isfinite(features).all():
            raise ValueError("Invalid audio features.")

        # Predict
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        probability_map = dict(
            zip(model.classes_, probabilities)
        )

        real_probability = float(probability_map.get(0, 0))
        ai_probability = float(probability_map.get(1, 0))

        label = "AI Generated" if prediction == 1 else "Real Speech"

        return {
            "filename": file.filename,
            "prediction": label,
            "real_probability": round(real_probability * 100, 2),
            "ai_probability": round(ai_probability * 100, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process audio: {str(e)}"
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()

