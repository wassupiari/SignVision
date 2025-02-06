import numpy as np
import tensorflow as tf
import cv2
from data_preprocessing import classes


def predict_image(model_path, image_path, img_size=(30, 30)):
    """Carica un modello salvato e fa una predizione su un'immagine."""
    # Caricare il modello
    model = tf.keras.models.load_model(model_path)

    # Caricare e preprocessare l'immagine
    img = cv2.imread(image_path)
    if img is None:
        print(f"Errore: impossibile caricare l'immagine {image_path}")
        return None

    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalizzazione

    # Predizione
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Restituzione del risultato
    return classes[predicted_class], confidence


# Esempio di utilizzo
if __name__ == "__main__":
    model_path = "models/traffic_signs_model.h5"
    image_path = "img_2.png"  # Modifica con il percorso dell'immagine
    prediction, confidence = predict_image(model_path, image_path)
    print(f"Predizione: {prediction} con confidenza {confidence:.2f}")