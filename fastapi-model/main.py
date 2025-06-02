from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

# Middleware CORS agar bisa diakses dari frontend/web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dari folder 'model'
model_path = os.path.join("model", "model_84.h5")
model = load_model(model_path)

# Fungsi preprocessing gambar
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Ukuran harus sesuai dengan saat training
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# Endpoint prediksi
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
        return JSONResponse(status_code=500, content={"error": str(e)})
