import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load MobileNetV2 dari ImageNet
mobilenet = MobileNetV2(weights="imagenet")

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
    img_resized = tf.image.resize(img_array, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_batch * 255)

    preds = mobilenet.predict(img_preprocessed)
    decoded = decode_predictions(preds, top=3)[0]

    for _, label, _ in decoded:
        label_lower = label.lower()
        for keyword, category in keyword_to_category.items():
            if keyword in label_lower:
                print(f"Gambar terdeteksi sebagai kategori: {category}")
                return category

    print("PERINGATAN: Gambar tidak dikenali.")
    return None
