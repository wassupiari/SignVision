import os
import shutil
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Percorsi cartelle
base_dir = Path("datasets")  # Cartella di origine
output_base_dir = Path("test_set")  # Cartella di destinazione
output_base_dir.mkdir(parents=True, exist_ok=True)

# Creazione sottocartelle per ogni classe
emotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
for emotion in emotion_classes:
    (output_base_dir / emotion).mkdir(parents=True, exist_ok=True)


# Funzione per copiare le immagini in base alla classe
def copy_images_by_class(src_folder, dest_folder, progress_bar):
    total_images = sum([len(files) for r, d, files in os.walk(src_folder) if
                        any(file.lower().endswith(('png', 'jpg', 'jpeg')) for file in files)])
    progress_bar['maximum'] = total_images
    progress_bar['value'] = 0
    progress_bar.update()

    for folder in os.listdir(src_folder):
        folder_path = src_folder / folder
        if folder_path.is_dir() and folder in emotion_classes:
            for img_file in os.listdir(folder_path):
                img_path = folder_path / img_file
                if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    try:
                        shutil.copy(img_path, dest_folder / folder / img_file)
                        print(f"Copia immagine: {img_path} -> {dest_folder / folder}")
                    except Exception as e:
                        print(f"Errore nel copiare {img_path}: {e}")
                    progress_bar['value'] += 1
                    progress_bar.update()


# Funzione principale
def build_test_set():
    root = tk.Tk()
    root.title("Creazione Test Set")
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress_bar.pack(padx=20, pady=20)

    copy_images_by_class(base_dir, output_base_dir, progress_bar)
    root.mainloop()
    print("Test set creato con successo!")


# Esegui la funzione
build_test_set()