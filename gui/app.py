from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Caricamento del modello
model_path = "models/traffic_signs_model_v2.h5"
model = tf.keras.models.load_model(model_path)

# Dizionario delle classi (preso da data_preprocessing.py)
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
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

# Rotta principale'to upload and predict
images@app.route('/', methods=['GET', 'POST'])
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
