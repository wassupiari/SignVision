from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Caricamento del modello
model_path = "models/traffic_signs_model_v3.h5"
model = tf.keras.models.load_model(model_path)

# Dizionario delle classi
classes = {
    0: 'Limite di velocità (20km/h)', 1: 'Limite di velocità (30km/h)', 2: 'Limite di velocità (50km/h)',
    3: 'Limite di velocità (60km/h)', 4: 'Limite di velocità (70km/h)', 5: 'Limite di velocità (80km/h)',
    6: 'Fine limite di velocità (80km/h)', 7: 'Limite di velocità (100km/h)', 8: 'Limite di velocità (120km/h)',
    9: 'Divieto di sorpasso', 10: 'Divieto di sorpasso per veicoli oltre 3.5 tonnellate', 11: 'Precedenza all incrocio',
    12: 'Strada con diritto di precedenza', 13: 'Dare precedenza', 14: 'Stop', 15: 'Divieto di transito', 16: 'Divieto per veicoli oltre 3.5 tonnellate',
    17: 'Divieto di accesso', 18: 'Pericolo generico', 19: 'Curva pericolosa a sinistra', 20: 'Curva pericolosa a destra',
    21: 'Doppia curva', 22: 'Strada dissestata', 23: 'Strada sdrucciolevole', 24: 'Strettoia sulla destra',
    25: 'Lavori in corso', 26: 'Semafori', 27: 'Attraversamento pedonale', 28: 'Attraversamento bambini',
    29: 'Attraversamento ciclisti', 30: 'Attenzione ghiaccio/neve', 31: 'Attraversamento animali selvatici',
    32: 'Fine limite di velocità e sorpasso', 33: 'Svolta a destra', 34: 'Svolta a sinistra',
    35: 'Dritto', 36: 'Dritto o destra', 37: 'Dritto o sinistra', 38: 'Tenere la destra',
    39: 'Tenere la sinistra', 40: 'Rotatoria obbligatoria', 41: 'Fine divieto di sorpasso', 42: 'Fine divieto di sorpasso per veicoli > 3.5 tonnellate'
}

# Configurazione di Flask
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Funzione di preprocessing per le immagini
def preprocess_image(image_path, target_size=(30, 30)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalizzazione
    return img

# Rotta principale
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocessa l'immagine e fai la predizione
            img = preprocess_image(file_path)
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            result = f"Predizione: {classes[predicted_class]} con confidenza {confidence:.2f}"

            return render_template('index.html', image=file_path, prediction=result)
    return render_template('index.html', image=None, prediction=None)

# Avvio dell'app Flask
if __name__ == '__main__':
    app.run(debug=True)