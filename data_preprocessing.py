import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_data(data_dir, csv_file, img_size=(30, 30)):
    """Carica il dataset e preprocessa le immagini."""
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for i in range(data.shape[0]):
        img_path = os.path.join(data_dir, data.iloc[i]['Path'])  # Colonna con il percorso dell'immagine

        if not os.path.exists(img_path):
            print(f"[WARNING] Immagine non trovata: {img_path}")
            continue  # Salta l'immagine

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Impossibile leggere l'immagine: {img_path}")
            continue  # Salta l'immagine non valida

        # Ritaglio ROI
        x1, y1, x2, y2 = data.iloc[i]['Roi.X1'], data.iloc[i]['Roi.Y1'], data.iloc[i]['Roi.X2'], data.iloc[i]['Roi.Y2']
        img = img[y1:y2, x1:x2]  # Ritaglio dell'immagine
        img = cv2.resize(img, img_size)  # Ridimensionamento

        images.append(img)
        labels.append(data.iloc[i]['ClassId'])  # Colonna con l'etichetta

    images = np.array(images) / 255.0  # Normalizzazione
    labels = to_categorical(np.array(labels))

    return images, labels


def split_data(images, labels, test_size=0.2, random_state=42):
    """Divide il dataset in training e validation set."""
    return train_test_split(images, labels, test_size=test_size, random_state=random_state, stratify=labels)
