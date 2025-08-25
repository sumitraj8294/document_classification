# src/models/infer.py
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Path to saved model
MODEL_PATH = os.path.join("models", "cnn_doc_classifier.h5")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (must match your train folder names)
CLASS_NAMES = ["Invoice", "Report", "Research"]   # change if you have more

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))   # same as training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return CLASS_NAMES[class_index], float(confidence)

if __name__ == "__main__":
    test_img = "data/test_images/3.png"   # put a sample test image here
    label, conf = predict_image(test_img)
    print(f"Predicted: {label} (confidence: {conf:.2f})")
