from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import gdown  # pastikan sudah install gdown di requirements.txt
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model_84.h5")
GDRIVE_URL = "https://drive.google.com/uc?id=1u7WfOS0SvGGtcjRSQDGjPfvjgI1D3HII"  # link unduhan Google Drive

# Fungsi untuk download model jika belum ada
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logging.info("Model tidak ditemukan, mulai download dari Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    else:
        logging.info("Model sudah ada, skip download.")

# Download model saat startup
download_model()

# Load model
model = load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)
        prediction = model.predict(img_array)[0][0]

        label = "layak" if prediction < 0.5 else "tidak_layak"
        confidence = float(prediction if label == "tidak_layak" else 1 - prediction)

        return JSONResponse(content={
            "label": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "API FastAPI berjalan"}
