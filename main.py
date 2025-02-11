from data_preprocessing import load_data, split_data, get_data_generators
from model import create_model
from training import train_model
from evaluation import evaluate_model
from plot_metrics import plot_training_history
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Configurazione
dir = "../MoodRipple/gtsrb-german-traffic-sign"
test_labels_path = "../MoodRipple/gtsrb-german-traffic-sign/Test.csv"
train_labels_path = "../MoodRipple/gtsrb-german-traffic-sign/Train.csv"
img_size = (30, 30)
epochs = 15
batch_size = 128
model_save_path = "models/15_epoche/model.h5"

def main():
    # Caricamento e preprocessing del dataset prima e dopo il bilanciamento
    train_images, train_labels = load_data(train_labels_path, dir, img_size)
    train_images, val_images, train_labels, val_labels = split_data(train_images, train_labels)

    test_images, test_labels = load_data(test_labels_path, dir, img_size)
    # Creazione dei generatori
    train_generator, val_generator = get_data_generators(train_images, train_labels, val_images, val_labels, batch_size)

    # Calcolo dei pesi delle classi per bilanciare il training
    y_train = np.argmax(train_labels, axis=1)
    num_classes = train_labels.shape[1]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
    class_weights = dict(enumerate(class_weights))

    print("[INFO] Pesi delle classi calcolati in main.py:", class_weights)

    # Creazione del modello
    model = create_model(input_shape=(img_size[0], img_size[1], 3), num_classes=train_labels.shape[1])

    # Addestramento del modello con pesi delle classi calcolati
    history = train_model(model, train_generator, val_generator, num_classes, epochs, class_weights=class_weights)

    # Salvataggio del modello addestrato
    model.save(model_save_path)
    print("Modello salvato con successo!")

    # Valutazione del modello
    evaluate_model(model, test_images, test_labels)

    # Grafici delle metriche
    plot_training_history(history, model, test_images, test_labels)

if __name__ == "__main__":
    main()
