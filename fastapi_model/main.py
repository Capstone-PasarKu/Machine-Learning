from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import logging

from model_validate import validate_market_image

app = FastAPI()

logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    interpreter = tf.lite.Interpreter(model_path="model/model_84.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("Model TFLite berhasil dimuat")
except Exception as e:
    logging.error(f"Gagal load model TFLite: {e}")
    raise e

class PredictionResult(BaseModel):
    label: str
    confidence: float
    category: str

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize untuk validasi kategori (224x224)
        img_for_validation = img.resize((224, 224))
        img_array_val = np.array(img_for_validation) / 255.0  # normalisasi 0-1
        
        category = validate_market_image(img_array_val)
        logging.info(f"Kategori gambar: {category}")

        if category == "unknown":
            return JSONResponse(
                status_code=400,
                content={"error": "Gambar tidak valid atau kategori tidak dikenali."}
            )

        # Resize untuk prediksi model TFLite (224x224)
        img_for_model = img.resize((224, 224))
        img_array_model = np.array(img_for_model) / 255.0
        img_array_model = np.expand_dims(img_array_model, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_array_model)
        interpreter.invoke()


        output_data = interpreter.get_tensor(output_details[0]['index'])
        value = output_data[0][0]

        if value >= 0.5:
            label = "tidak layak"
            confidence = value
        else:
            label = "layak"
            confidence = 1 - value

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "category": category
        }

    except Exception as e:
        logging.error(f"Error pada prediksi: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
