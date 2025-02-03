import os
import numpy as np
import shutil
from sklearn.decomposition import PCA
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# 📌 Percorsi delle cartelle
dataset_path = "datasets/fer2013/train"  # Cartella originale con le immagini
output_dir = "datasets/fer2013/selected_images"  # Dove salvare le immagini selezionate
os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste

# 📌 Carica il modello ResNet50 pre-addestrato
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    """Estrai le feature da un'immagine utilizzando ResNet50."""
    img = image.load_img(img_path, target_size=(48, 48))  # FER2013 è 96x96
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# 📌 Estrazione delle feature da tutte le immagini
features = []
image_paths = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(root, file)
            img_features = extract_features(img_path, model)
            features.append(img_features)
            image_paths.append(img_path)

features = np.array(features)

# 📌 Applica PCA per ridurre la dimensionalità mantenendo il 95% della varianza
pca = PCA(n_components=0.95)  # Mantiene il 95% della varianza
pca_features = pca.fit_transform(features)

# 📌 Calcola la varianza per ogni immagine
variances = np.var(pca_features, axis=1)

# 📌 Ordina le immagini per varianza decrescente
sorted_indices = np.argsort(variances)[::-1]

# 📌 Seleziona le immagini più rappresentative
num_images_to_select = 100  # Modifica questo valore in base alle tue esigenze
selected_indices = sorted_indices[:num_images_to_select]
selected_images = [image_paths[i] for i in selected_indices]

# 📌 Copia le immagini selezionate nella cartella di output
for img_path in selected_images:
    shutil.copy(img_path, output_dir)

print(f"✅ Selezionate {len(selected_images)} immagini con la massima varianza e salvate in {output_dir}")
