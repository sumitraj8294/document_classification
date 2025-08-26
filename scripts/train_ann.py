import os
import tensorflow as tf
from src.ingest.ocr import ocr_image
from src.models.ann_text_classifier import ANNTextClassifier

import pickle

# Example training dataset (update with your own)
CLASS_NAMES = ["Invoice", "Report", "Research"]

train_dir = "data/train_images"
classes = os.listdir(train_dir)

texts = []
labels = []

for idx, cls in enumerate(classes):
    cls_dir = os.path.join(train_dir, cls)
    for img_file in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, img_file)
        text = ocr_image(img_path)

        texts.append(text)
        labels.append(idx)

classifier = ANNTextClassifier(vocab_size=5000, maxlen=100)
classifier.train(texts, labels, epochs=5, batch_size=16)

# Save model
classifier.model.save("ann_model.h5")
