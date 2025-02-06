import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurazione percorsi
model_path = "models/model_2.h5"  # Percorso del modello salvato
test_data_path = "Test"  # Percorso del test set
img_width, img_height = 30, 30  # Dimensioni delle immagini
batch_size = 32  # Dimensione batch

# Caricare il modello addestrato
model = tf.keras.models.load_model(model_path)
print("Modello caricato con successo!")

# Creare un data generator per il test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Manteniamo l'ordine per la valutazione
)

# Predizioni
y_pred_probs = model.predict(test_generator)  # Probabilità per ogni classe
y_pred = np.argmax(y_pred_probs, axis=1)  # Etichette predette
y_true = test_generator.classes  # Etichette reali

# Classi del dataset
class_labels = list(test_generator.class_indices.keys())

# Stampare il report di classificazione
print("\n\U0001F4CA Report di Classificazione:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Matrice di confusione
cm = confusion_matrix(y_true, y_pred)

# Visualizzazione della matrice di confusione
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione")
plt.show()

# Funzione per prevedere il segnale stradale da un'immagine
def predict_traffic_sign(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizzazione

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return class_labels[predicted_class]

# Test su un'immagine specifica
test_image = "resized_image.png"  # Sostituisci con il percorso reale
detected_sign = predict_traffic_sign(test_image)
print(f"Il segnale stradale predetto per {test_image} è: {detected_sign}")