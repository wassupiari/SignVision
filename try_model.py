import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from data_preprocessing import classes


def predict_images_in_folder(model_path, folder_path, csv_path, img_size=(30, 30)):
    """Carica un modello salvato, fa predizioni su tutte le immagini in una cartella e confronta con le etichette nel CSV."""
    model = tf.keras.models.load_model(model_path)

    # Caricare il file CSV con le etichette reali
    labels_df = pd.read_csv(csv_path)
    labels_df['Path'] = labels_df['Path'].apply(lambda x: os.path.basename(x))  # Estrarre solo il nome del file
    labels_dict = dict(zip(labels_df['Path'], labels_df['ClassId']))

    predictions = []
    correct = 0
    total = 0

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Immagine non trovata o non valida: {img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=0) / 255.0  # Normalizzazione

        pred_probs = model.predict(img)
        predicted_class = np.argmax(pred_probs)
        confidence = np.max(pred_probs)

        real_class = labels_dict.get(filename, None)
        correct_prediction = predicted_class == real_class

        predictions.append((filename, classes[predicted_class], confidence, correct_prediction))

        if correct_prediction:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuratezza complessiva: {accuracy:.2%}")

    return predictions


# Esempio di utilizzo
if __name__ == "__main__":
    model_path = "models/traffic_signs_model_v3.h5"
    folder_path = "../SignVision/gtsrb-german-traffic-sign/Meta/"  # Modifica con il percorso della cartella
    csv_path = "../SignVision/gtsrb-german-traffic-sign/Meta.csv"  # Modifica con il percorso del CSV
    results = predict_images_in_folder(model_path, folder_path, csv_path)

    for filename, prediction, confidence, correct in results:
        correctness = "✅" if correct else "❌"
        print(f"{filename}: {prediction} (Confidenza: {confidence:.2f}) {correctness}")
