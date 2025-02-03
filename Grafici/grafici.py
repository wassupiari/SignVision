import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Imposta il percorso della cartella del dataset
dataset_path = "../datasets/fer2013/custom_train"

# Legge le classi del dataset (nomi delle sottocartelle)
classes = os.listdir(dataset_path)
class_counts = {}

# Conta le immagini per ogni classe
for class_name in classes:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        class_counts[class_name] = len(os.listdir(class_dir))

# Visualizza la distribuzione delle classi
plt.figure(figsize=(10, 5))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.xticks(rotation=45)
plt.xlabel("Classe")
plt.ylabel("Numero di immagini")
plt.title("Distribuzione delle immagini per classe")
plt.show()

# Analisi delle dimensioni delle immagini
image_sizes = []
for class_name in classes:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir)[:50]:  # Prende un campione di 50 immagini per classe
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                image_sizes.append(img.shape[:2])  # (altezza, larghezza)

# Converte in un array per il grafico
image_sizes = np.array(image_sizes)
plt.figure(figsize=(8, 5))
plt.scatter(image_sizes[:, 1], image_sizes[:, 0], alpha=0.5, color='b')
plt.xlabel("Larghezza")
plt.ylabel("Altezza")
plt.title("Distribuzione delle dimensioni delle immagini")
plt.show()

# Mostra alcune immagini campione
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    class_name = np.random.choice(classes)
    img_name = np.random.choice(os.listdir(os.path.join(dataset_path, class_name)))
    img_path = os.path.join(dataset_path, class_name, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converti da BGR a RGB per matplotlib
    ax.imshow(img)
    ax.set_title(class_name)
    ax.axis("off")

plt.tight_layout()
plt.show()