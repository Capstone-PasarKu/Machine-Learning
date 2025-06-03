from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import logging

from model_validate import validate_market_image  # pastikan file ini ada

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model dari folder model/
interpreter = tf.lite.Interpreter(model_path="model/model_84.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Response schema
class PredictionResult(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Validasi isi gambar
        category = validate_market_image(img_array[0])
        if category is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Gambar tidak valid."}
            )

        # Prediksi dengan model tflite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        value = output_data[0][0]

        label = "layak" if value < 0.5 else "tidak layak"
        confidence = float(value if label == "tidak layak" else 1 - value)

        return {"label": label, "confidence": round(confidence, 4)}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
