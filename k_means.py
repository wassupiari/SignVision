import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
output_path = "k_means"
grafici_dir = "grafici"
os.makedirs(grafici_dir, exist_ok=True)

# Definizione dimensione immagine
TARGET_SIZE = (96, 96)

# Lista per memorizzare feature e immagini
features_list = []
image_paths = []

# Scandire le cartelle nel dataset
for emotion_folder in os.listdir(dataset_path):
    emotion_path = os.path.join(dataset_path, emotion_folder)
    output_emotion_path = os.path.join(output_path, emotion_folder)

    if not os.path.isdir(emotion_path):
        continue

    os.makedirs(output_emotion_path, exist_ok=True)

    for img_name in tqdm(os.listdir(emotion_path), desc=f"Processing {emotion_folder}"):
        img_path = os.path.join(emotion_path, img_name)

        img = image.load_img(img_path, target_size=TARGET_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalizzazione

        # Appiattire l'immagine per usarla come feature
        features_list.append(img_array.flatten())
        image_paths.append(img_path)

# Convertire le feature in array numpy
features_matrix = np.array(features_list)

# Applicare PCA per ridurre le feature a 50 dimensioni
pca = PCA(n_components=50)
features_reduced = pca.fit_transform(features_matrix)

# Applicare K-Means Clustering
num_clusters = 8  # Numero di cluster (corrisponde alle emozioni)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features_reduced)

# Selezionare immagini rappresentative per ciascun cluster
selected_images = []
for i in range(num_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    centroid = kmeans.cluster_centers_[i]
    distances = [np.linalg.norm(features_reduced[j] - centroid) for j in cluster_indices]
    best_image = cluster_indices[np.argmin(distances)]
    selected_images.append(image_paths[best_image])

# Creare grafico della distribuzione con K-Means
plt.figure(figsize=(8, 6))
plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.5, marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label="Centroidi")
plt.xlabel("Componente principale 1")
plt.ylabel("Componente principale 2")
plt.title("Distribuzione delle immagini selezionate con K-Means")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(grafici_dir, "distribuzione_kmeans.png"))
plt.show()

print("Selezione completata. Le immagini rappresentative per ogni cluster sono state salvate.")
