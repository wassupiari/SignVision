from flask import Flask, request, render_template, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2

# Inizializza Flask
app = Flask(__name__)

# Configurazione della cartella static per salvare le immagini
UPLOAD_FOLDER = "../MoodRipple/gui/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Caricamento del modello
model_path = "models/traffic_signs_model_v7.h5"
model = tf.keras.models.load_model(model_path)

# Dizionario delle classi
classes = {
    0: 'Limite di velocità (20km/h)', 1: 'Limite di velocità (30km/h)', 2: 'Limite di velocità (50km/h)',
    3: 'Limite di velocità (60km/h)', 4: 'Limite di velocità (70km/h)', 5: 'Limite di velocità (80km/h)',
    6: 'Fine limite di velocità (80km/h)', 7: 'Limite di velocità (100km/h)', 8: 'Limite di velocità (120km/h)',
    9: 'Divieto di sorpasso', 10: 'Divieto di sorpasso per veicoli oltre 3.5 tonnellate', 11: 'Precedenza all\'incrocio',
    12: 'Strada con diritto di precedenza', 13: 'Dare precedenza', 14: 'Stop', 15: 'Divieto di transito', 16: 'Divieto per veicoli oltre 3.5 tonnellate',
    17: 'Divieto di accesso', 18: 'Pericolo generico', 19: 'Curva pericolosa a sinistra', 20: 'Curva pericolosa a destra',
    21: 'Doppia curva', 22: 'Strada dissestata', 23: 'Strada sdrucciolevole', 24: 'Strettoia sulla destra',
    25: 'Lavori in corso', 26: 'Semafori', 27: 'Attraversamento pedonale', 28: 'Attraversamento bambini',
    29: 'Attraversamento ciclisti', 30: 'Attenzione ghiaccio/neve', 31: 'Attraversamento animali selvatici',
    32: 'Fine limite di velocità e sorpasso', 33: 'Svolta a destra', 34: 'Svolta a sinistra',
    35: 'Dritto', 36: 'Dritto o destra', 37: 'Dritto o sinistra', 38: 'Tenere la destra',
    39: 'Tenere la sinistra', 40: 'Rotatoria obbligatoria', 41: 'Fine divieto di sorpasso', 42: 'Fine divieto di sorpasso per veicoli > 3.5 tonnellate'
}

# Funzione di preprocessing per le immagini
def preprocess_image(image_path, target_size=(30, 30)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalizzazione
    return img

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Percorso dell'immagine per il rendering in HTML
            image_url = url_for("static", filename=f"uploads/{filename}") + f"?v={int(os.path.getmtime(file_path))}"

            # Preprocessa l'immagine e fai la predizione
            img = preprocess_image(file_path)
            predictions = model.predict(img)[0]  # Ottieni il vettore di probabilità per ogni classe
            percentages = (predictions * 100).round(2)  # Converti in percentuale

            # Classe più probabile
            top_prediction = max(enumerate(percentages), key=lambda x: x[1])
            best_label = classes[top_prediction[0]]
            best_confidence = top_prediction[1]

            # Tutte le predizioni ordinate
            all_predictions = sorted(
                [(classes[i], percentages[i]) for i in range(len(classes))],
                key=lambda x: x[1], reverse=True
            )

            return render_template(
                "index.html", image=image_url, best_prediction=(best_label, best_confidence), all_predictions=all_predictions
            )

    return render_template("index.html", image=None, best_prediction=None, all_predictions=None)



if __name__ == "__main__":
    app.run(debug=True)
