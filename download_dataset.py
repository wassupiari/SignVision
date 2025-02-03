# Funzione per scaricare il dataset da kaggle e salvarlo in una cartella
import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split

# Percorso della cartella dove sta il file ZIP
dataset_zip = "archive.zip"
extract_path = "datasets/fer2013"

# Crea la cartella di destinazione se non esiste
os.makedirs(extract_path, exist_ok=True)

# Estrai il contenuto del file ZIP
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Dataset estratto in: {extract_path}")

# Percorsi delle cartelle originali di train e test
train_dir = "datasets/fer2013/train"
test_dir = "datasets/fer2013/test"

# Nuova cartella per i dati combinati
all_data_dir = "datasets/fer2013/all_data"
os.makedirs(all_data_dir, exist_ok=True)

# Unisci le immagini da train e test in un'unica cartella
for category in os.listdir(train_dir):
    category_path = os.path.join(train_dir, category)
    new_category_path = os.path.join(all_data_dir, category)
    os.makedirs(new_category_path, exist_ok=True)

    for file in os.listdir(category_path):
        shutil.copy(os.path.join(category_path, file), new_category_path)

for category in os.listdir(test_dir):
    category_path = os.path.join(test_dir, category)
    new_category_path = os.path.join(all_data_dir, category)
    os.makedirs(new_category_path, exist_ok=True)

    for file in os.listdir(category_path):
        shutil.copy(os.path.join(category_path, file), new_category_path)

print("Tutti i dati sono stati uniti nella cartella:", all_data_dir)

# Creazione delle nuove cartelle train e test
final_train_dir = "datasets/fer2013/custom_train"
final_test_dir = "datasets/fer2013/custom_test"

for folder in [final_train_dir, final_test_dir]:
    os.makedirs(folder, exist_ok=True)

# Suddivisione in train/test (es. 80% train, 20% test)
for category in os.listdir(all_data_dir):
    category_path = os.path.join(all_data_dir, category)
    images = os.listdir(category_path)

    # Dividi in train (80%) e test (20%)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Creazione delle sottocartelle nelle cartelle train e test
    os.makedirs(os.path.join(final_train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(final_test_dir, category), exist_ok=True)

    # Sposta le immagini nella rispettiva cartella
    for img in train_images:
        shutil.move(os.path.join(category_path, img), os.path.join(final_train_dir, category, img))

    for img in test_images:
        shutil.move(os.path.join(category_path, img), os.path.join(final_test_dir, category, img))

print(f"Dataset suddiviso: {final_train_dir} (train) e {final_test_dir} (test)")

