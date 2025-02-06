import os
import zipfile
import kagglehub

# Percorso della cartella del progetto
project_path = os.path.dirname(os.path.abspath(__file__))

def download_and_extract_dataset(dataset_name, extract_path):
    print(f"Downloading dataset {dataset_name}...")
    dataset_path = kagglehub.dataset_download(dataset_name, path=extract_path)
    print(f"Dataset scaricato in: {dataset_path}")

# Nome del dataset su KaggleHub
dataset_name = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"

# Percorso della cartella di destinazione
extract_to_path = os.path.join(project_path, "GTSRB")

# Scaricare ed estrarre il dataset
download_and_extract_dataset(dataset_name, extract_to_path)
