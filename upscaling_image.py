import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from realesrgan import RealESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet

# Percorsi delle cartelle
dataset_path = "datasets"
output_path = "datasets_upscaled"

# Verifica se la GPU è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Utilizzando il dispositivo: {device}")

# Carica il modello Real-ESRGAN
model = RealESRGAN(device, scale=4)
model.load_weights("weights/RealESRGAN_x4plus.pth")
model.to(device).eval()

# Creazione cartella di output
os.makedirs(output_path, exist_ok=True)

# Elaborazione delle immagini con ESRGAN
for emotion_folder in os.listdir(dataset_path):
    emotion_path = os.path.join(dataset_path, emotion_folder)
    output_emotion_path = os.path.join(output_path, emotion_folder)

    if not os.path.isdir(emotion_path):
        continue  # Salta file non cartelle

    os.makedirs(output_emotion_path, exist_ok=True)

    for img_name in tqdm(os.listdir(emotion_path), desc=f"Processing {emotion_folder}"):
        img_path = os.path.join(emotion_path, img_name)
        output_img_path = os.path.join(output_emotion_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue  # Salta immagini non valide

        # Converti in RGB per ESRGAN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = torch.from_numpy(img_rgb).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)

        # Applica ESRGAN per l'upscaling
        with torch.no_grad():
            upscaled_img = model.predict(img_rgb)

        # Converti l'immagine migliorata in formato OpenCV
        upscaled_img = (upscaled_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Ridimensiona a 224x224 per il training
        upscaled_img = cv2.resize(upscaled_img, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Salva l'immagine migliorata
        cv2.imwrite(output_img_path, cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2BGR))

print("✅ Preprocessing completato con ESRGAN: immagini migliorate e ridimensionate a 224x224.")
