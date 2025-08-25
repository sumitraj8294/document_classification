# scripts/train_cnn.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths
BASE_DIR = os.path.join("data", "train_images")
MODEL_SAVE_PATH = os.path.join("models", "cnn_doc_classifier.h5")

# Hyperparameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5   # keep small for now

# Data pipeline
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved at {MODEL_SAVE_PATH}")
