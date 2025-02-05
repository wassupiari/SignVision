import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Configurazione GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Utilizzando la GPU.")
else:
    print("Nessuna GPU trovata, si utilizza la CPU.")

# Percorsi
dataset_path = "datasets"
output_path = "varianza"
grafici_dir = "grafici"
os.makedirs(grafici_dir, exist_ok=True)

# Definizione dimensione immagine
TARGET_SIZE = (96, 96)

# Caricare MobileNetV2 pre-addestrata
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Lista per memorizzare la varianza delle immagini
image_variances = []
features_list = []
image_paths = []

# Scandire le cartelle nel dataset
for emotion_folder in os.listdir(dataset_path):
    emotion_path = os.path.join(dataset_path, emotion_folder)
    output_emotion_path = os.path.join(output_path, emotion_folder)

    if not os.path.isdir(emotion_path):
        continue  # Saltiamo file non cartelle

    os.makedirs(output_emotion_path, exist_ok=True)

    image_data = []
    for img_name in tqdm(os.listdir(emotion_path), desc=f"Processing {emotion_folder}"):
        img_path = os.path.join(emotion_path, img_name)

        img = image.load_img(img_path, target_size=TARGET_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Estrazione feature con MobileNetV2
        features = model.predict(img_array, verbose=0)
        features_list.append(features.flatten())
        image_paths.append(img_path)

        # Calcolo varianza delle feature
        variance = np.var(features)
        image_data.append((img_path, variance))

    # Creazione DataFrame con varianze
    df_variance = pd.DataFrame(image_data, columns=["image_path", "variance"])

    # Selezionare le prime 2500 immagini con la varianza più alta
    df_top2500 = df_variance.sort_values(by="variance", ascending=False).head(2500)

    # Copiare le immagini selezionate nella nuova cartella
    for img_path in tqdm(df_top2500["image_path"], desc=f"Copying top images for {emotion_folder}"):
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(output_emotion_path, img_name)
        cv2.imwrite(output_img_path, cv2.imread(img_path))

    # Salvare lista immagini selezionate
    selected_images_csv = os.path.join(output_emotion_path, "selected_images.csv")
    df_top2500.to_csv(selected_images_csv, index=False)

# Applicare PCA per ridurre le feature a 2 dimensioni
features_matrix = np.array(features_list)
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(features_matrix)

# Creare grafico della distribuzione delle immagini selezionate
plt.figure(figsize=(8, 6))
plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c='blue', alpha=0.5, marker='o',
            label="Immagini selezionate")
plt.xlabel("Componente principale 1")
plt.ylabel("Componente principale 2")
plt.title("Distribuzione delle immagini selezionate con PCA")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(grafici_dir, "distribuzione_pca.png"))
plt.show()

print("Selezione completata. Le immagini con la varianza più alta sono state salvate.")
