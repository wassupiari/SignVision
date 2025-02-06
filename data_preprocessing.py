import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Dizionario delle classi
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


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
