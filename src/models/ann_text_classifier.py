import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

class ANNTextClassifier:
    def __init__(self, vocab_size=5000, maxlen=100):
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.model = None

    def preprocess_texts(self, texts, fit_tokenizer=False):
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.maxlen, padding='post', truncating='post')
        return padded

    def build_model(self, num_classes):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.maxlen,)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model

    def train(self, texts, labels, epochs=10, batch_size=32):
        X = self.preprocess_texts(texts, fit_tokenizer=True)
        y = np.array(labels)
        num_classes = len(set(labels))
        self.build_model(num_classes)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        # save tokenizer
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

    def predict(self, texts):
        X = self.preprocess_texts(texts, fit_tokenizer=False)
        predictions = self.model.predict(X)
        return predictions.argmax(axis=1)
