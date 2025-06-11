import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import logging

logging.basicConfig(level=logging.INFO)

# Load MobileNetV2 dengan weights ImageNet
try:
    mobilenet = MobileNetV2(weights="imagenet")
    logging.info("MobileNetV2 berhasil dimuat.")
except Exception as e:
    logging.error(f"Gagal memuat MobileNetV2: {e}")
    raise e

keyword_to_category = {
    'meat': 'meat', 'beef': 'meat', 'pork': 'meat', 'chicken': 'meat', 'steak': 'meat',
    'lamb': 'meat', 'duck': 'meat', 'turkey': 'meat', 'crab': 'seafood', 'shrimp': 'seafood',
    'fish': 'seafood', 'salmon': 'seafood', 'tuna': 'seafood', 'fruit': 'fruit', 'apple': 'fruit',
    'banana': 'fruit', 'orange': 'fruit', 'grape': 'fruit', 'melon': 'fruit', 'mango': 'fruit',
    'strawberry': 'fruit', 'pineapple': 'fruit', 'lemon': 'fruit', 'fig': 'fruit',
    'vegetable': 'vegetable', 'carrot': 'vegetable', 'cabbage': 'vegetable', 'broccoli': 'vegetable',
    'cauliflower': 'vegetable', 'spinach': 'vegetable', 'cucumber': 'vegetable',
    'zucchini': 'vegetable', 'pepper': 'vegetable', 'bell_pepper': 'vegetable', 'lettuce': 'vegetable',
    'corn': 'vegetable', 'garlic': 'spice', 'onion': 'spice', 'ginger': 'spice',
    'shallot': 'spice', 'spice': 'spice', 'herb': 'spice',
}

def validate_market_image(img_array):
    try:
        # Pastikan input adalah float32 numpy array dengan shape (224,224,3)
        if not isinstance(img_array, np.ndarray):
            logging.warning("Input bukan numpy array")
            return "unknown"
        if img_array.shape != (224, 224, 3):
            logging.warning(f"Input shape tidak sesuai: {img_array.shape}")
            return "unknown"

        img_resized = tf.image.resize(img_array, (224, 224))
        img_batch = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_batch * 255.0)

        preds = mobilenet.predict(img_preprocessed)
        decoded = decode_predictions(preds, top=5)[0]

        logging.info(f"Hasil prediksi top-5: {decoded}")

        for _, label, prob in decoded:
            label_lower = label.lower()
            for keyword, category in keyword_to_category.items():
                if keyword in label_lower:
                    logging.info(f"Gambar dikenali sebagai: {label_lower}, kategori: {category}")
                    return category

        logging.info("Kategori tidak dikenali, fallback ke 'unknown'")
        return "unknown"

    except Exception as e:
        logging.error(f"Error saat validasi gambar: {e}")
        return "unknown"
