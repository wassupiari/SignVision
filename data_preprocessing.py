import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Dizionario delle classi
classes = {
    0: 'Limite di velocità (20km/h)', 1: 'Limite di velocità (30km/h)', 2: 'Limite di velocità (50km/h)',
    3: 'Limite di velocità (60km/h)', 4: 'Limite di velocità (70km/h)', 5: 'Limite di velocità (80km/h)',
    6: 'Fine limite di velocità (80km/h)', 7: 'Limite di velocità (100km/h)', 8: 'Limite di velocità (120km/h)',
    9: 'Divieto di sorpasso', 10: 'Divieto di sorpasso per veicoli oltre 3.5 tonnellate', 11: 'Precedenza all incrocio',
    12: 'Strada con diritto di precedenza', 13: 'Dare precedenza', 14: 'Stop', 15: 'Divieto di transito', 16: 'Divieto per veicoli oltre 3.5 tonnellate',
    17: 'Divieto di accesso', 18: 'Pericolo generico', 19: 'Curva pericolosa a sinistra', 20: 'Curva pericolosa a destra',
    21: 'Doppia curva', 22: 'Strada dissestata', 23: 'Strada sdrucciolevole', 24: 'Strettoia sulla destra',
    25: 'Lavori in corso', 26: 'Semafori', 27: 'Attraversamento pedonale', 28: 'Attraversamento bambini',
    29: 'Attraversamento ciclisti', 30: 'Attenzione ghiaccio/neve', 31: 'Attraversamento animali selvatici',
    32: 'Fine limite di velocità e sorpasso', 33: 'Svolta a destra', 34: 'Svolta a sinistra',
    35: 'Dritto', 36: 'Dritto o destra', 37: 'Dritto o sinistra', 38: 'Tenere la destra',
    39: 'Tenere la sinistra', 40: 'Rotatoria obbligatoria', 41: 'Fine divieto di sorpasso', 42: 'Fine divieto di sorpasso per veicoli > 3.5 tonnellate'
}

def load_data(csv_file, data_dir, img_size=(30, 30)):
    """Carica il dataset e preprocessa le immagini."""
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for i in range(data.shape[0]):
        img_path = os.path.normpath(os.path.join(data_dir, str(data.iloc[i]['Path']).strip().replace('Train/Train/', 'Train/').replace('Test/Test/', 'Test/'))).replace('\'', '/')

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
    labels = to_categorical(np.array(labels), num_classes=len(classes))

    return images, labels


def split_data(images, labels, test_size=0.2, random_state=42):
    """Divide il dataset in training e validation set."""
    return train_test_split(images, labels, test_size=test_size, random_state=random_state, stratify=labels)
