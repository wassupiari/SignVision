import os
import shutil
import numpy as np
import gc
import tensorflow as tf
from sklearn.decomposition import PCA
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# 📌 Percorsi delle cartelle
source_dir = "datasets/fer2013/train"  # Dataset originale
output_dir = "datasets/fer2013/filtered_images"  # Dove salveremo il dataset filtrato
grafici_dir = "grafici"  # Dove salveremo i grafici
os.makedirs(output_dir, exist_ok=True)
os.makedirs(grafici_dir, exist_ok=True)

# 📌 Configura TensorFlow per usare la GPU su Mac M2
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("✅ TensorFlow sta usando la GPU!")
else:
    print("⚠ TensorFlow non ha trovato una GPU, verrà usata la CPU.")

# 📌 Carica il modello ResNet50 pre-addestrato
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def load_image(img_path):
    """Carica un'immagine e la pre-processa per ResNet50."""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((48, 48))  # FER2013 è 96x96
        img_array = np.array(img)
        img_array = preprocess_input(img_array)  # Normalizzazione per ResNet50
        return img_array
    except Exception as e:
        print(f"❌ Errore caricamento immagine {img_path}: {e}")
        return None


def load_images(folder):
    """Carica immagini da tutte le sottocartelle della cartella principale."""
    images, paths, labels = [], [], []

    print(f"🔍 Scansionando cartella: {folder}")

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)

        if os.path.isdir(class_path):  # Se è una cartella
            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)

                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_array = load_image(img_path)

                    if img_array is not None:
                        images.append(img_array)
                        paths.append(img_path)
                        labels.append(class_name)  # Salva la classe dell'immagine

    print(f"✅ Trovate {len(paths)} immagini in {folder}")  # Debug
    return np.array(images), paths, labels


def extract_features(images):
    """Estrai feature da immagini usando ResNet50 con accelerazione GPU."""
    images = tf.convert_to_tensor(images)  # Convertiamo in tensori TensorFlow
    features = model.predict(images, batch_size=32)  # Estraggo feature con batch_size ottimizzata
    return features


def filter_images(images, paths, labels):
    """Filtra immagini in base alla varianza delle feature PCA."""
    # 📌 Estraggo le feature con ResNet50
    features = extract_features(images)

    # 📌 Applico PCA per ridurre la dimensionalità
    pca = PCA(n_components=50)
    features_reduced = pca.fit_transform(features)

    # 📌 Calcolo la varianza delle feature
    variances = np.var(features_reduced, axis=1)

    # 📌 Imposto una soglia dinamica: elimino le immagini con bassa varianza (10° percentile)
    min_variance_threshold = np.percentile(variances, 10)
    selected_indices = [i for i, var in enumerate(variances) if var >= min_variance_threshold]

    selected_paths = [paths[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]  # Mantiene la classe di ogni immagine

    return selected_paths, selected_labels, features_reduced, selected_indices


def save_variance_plot(features_reduced, selected_indices):
    """Salva il grafico della distribuzione della varianza delle immagini."""
    plt.figure(figsize=(8, 6))

    # Riduzione a 2D per visualizzazione
    pca_2d = PCA(n_components=2)
    reduced_2d = pca_2d.fit_transform(features_reduced)

    # Grafico dispersione
    plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c='blue', marker='o', alpha=0.5, label='Tutte le immagini')

    # Evidenzio le immagini selezionate
    selected_2d = reduced_2d[selected_indices]
    plt.scatter(selected_2d[:, 0], selected_2d[:, 1], c='red', marker='x', s=100, label='Immagini selezionate')

    plt.title("Selezione immagini basata sulla varianza")
    plt.xlabel("Componente principale 1")
    plt.ylabel("Componente principale 2")
    plt.legend()
    plt.savefig(os.path.join(grafici_dir, "selezione_variance.png"))
    plt.close()


def copy_selected_images(selected_paths, selected_labels):
    """Copia le immagini selezionate mantenendo la struttura originale delle cartelle."""
    for img_path, class_name in zip(selected_paths, selected_labels):
        class_output_dir = os.path.join(output_dir, class_name)  # Mantiene la cartella della classe
        os.makedirs(class_output_dir, exist_ok=True)  # Crea la cartella se non esiste

        try:
            shutil.copy(img_path, os.path.join(class_output_dir, os.path.basename(img_path)))
        except Exception as e:
            print(f"❌ Errore copia immagine {img_path}: {e}")


def process_dataset():
    """Processa tutto il dataset, rimuovendo immagini ridondanti con varianza bassa."""
    print(f"📂 Inizio elaborazione dataset: {source_dir}")

    images, paths, labels = load_images(source_dir)
    if not images.size:
        print("❌ Nessuna immagine trovata.")
        return

    selected_paths, selected_labels, features_reduced, selected_indices = filter_images(images, paths, labels)

    # Salva il grafico delle immagini selezionate
    save_variance_plot(features_reduced, selected_indices)

    # Copia le immagini selezionate nella cartella finale, mantenendo la struttura originale
    copy_selected_images(selected_paths, selected_labels)

    # Pulizia memoria
    del images, paths, labels
    gc.collect()
    tf.keras.backend.clear_session()

    print(f"✅ Processo completato. Immagini selezionate salvate in: {output_dir}")
    print(f"📊 Grafico della selezione salvato in: {grafici_dir}")


# 🚀 Avvia il processo
process_dataset()
