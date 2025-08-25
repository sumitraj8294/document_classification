import joblib
import os
from tensorflow.keras.models import load_model
import numpy as np

VEC_PATH = "models/tfidf.pkl"
ANN_PATH = "models/ann_text_classifier.h5"
CATEGORIES = ["Finance", "Legal", "Research"]

def predict_text_category(text: str) -> str:
    if not (os.path.exists(VEC_PATH) and os.path.exists(ANN_PATH)):
        # fallback if model not trained yet
        return "Research"
    vec = joblib.load(VEC_PATH)
    model = load_model(ANN_PATH)
    X = vec.transform([text]).toarray()
    pred = model.predict(X, verbose=0)
    return CATEGORIES[int(np.argmax(pred))]
