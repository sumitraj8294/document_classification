import os
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from src.ingest.ocr import ocr_image

# === Load ANN + OCR ===
CLASS_NAMES = ["Invoice", "Report", "Research"]
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

ann_model = tf.keras.models.load_model("models/ann_model.h5")

MAXLEN = 100  # must match training

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAXLEN, padding='post', truncating='post')
    return padded

def predict_with_ann(img_path):
    text = ocr_image(img_path)
    print("\n📄 OCR Extracted Text (first 300 chars):")
    print(text[:300] + ("..." if len(text) > 300 else ""))

    X = preprocess_text(text)
    prediction = ann_model.predict(X)
    return np.argmax(prediction, axis=1)[0]

# === Load CNN ===
cnn_model = tf.keras.models.load_model("models/cnn_doc_classifier.h5")

def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size, color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_with_cnn(img_path):
    X = preprocess_image(img_path)
    prediction = cnn_model.predict(X)
    return np.argmax(prediction, axis=1)[0]

# === Main inference ===
# Absolute base path for safety
BASE_DIR = r"C:\Users\sumit raj\Desktop\document_classification"

if __name__ == "__main__":
    # Ask user for image path
    user_input = input("Enter image path (press Enter to use default test image): ").strip()

    if user_input:
        # If user provides absolute path, use directly
        if os.path.isabs(user_input):
            img_path = user_input
        else:
            # If user provides just filename, assume inside data/test_images
            img_path = os.path.join(BASE_DIR, "data", "test_images", user_input)
    else:
        # Default fallback
        img_path = os.path.join(BASE_DIR, "data", "test_images", "research.png")

    # Validate path
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        exit(1)

    print(f"✅ Using image: {img_path}")

    # Run CNN
    cnn_class = predict_with_cnn(img_path)
    print(f"[CNN Prediction] {cnn_class}")

    # Run ANN + OCR
    ann_class = predict_with_ann(img_path)
    print(f"[ANN + OCR Prediction] {ann_class}")
    print(f"[CNN Prediction] {cnn_class} ({CLASS_NAMES[cnn_class]})")
print(f"[ANN + OCR Prediction] {ann_class} ({CLASS_NAMES[ann_class]})")
