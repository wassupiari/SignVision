from data_preprocessing import load_data, split_data
from model import create_model
from training import train_model
from evaluation import evaluate_model
from plot_metrics import plot_training_history
import os

# Configurazione
train_data_path = "../MoodRipple"
test_data_path = "../MoodRipple"
test_labels_path = "Dataset/Test.csv"
train_labels_path = "Dataset/Train.csv"
img_size = (30, 30)
epochs = 15
batch_size = 32
model_save_path = "models/traffic_signs_model_v3.h5"

def main():
    # Caricamento e preprocessing del dataset
    train_images, train_labels = load_data(train_labels_path, train_data_path, img_size)
    train_images, val_images, train_labels, val_labels = split_data(train_images, train_labels)

    # Caricamento e preprocessing del test set
    test_images, test_labels = load_data(test_labels_path, test_data_path, img_size)

    # Creazione del modello
    model = create_model(input_shape=(img_size[0], img_size[1], 3), num_classes=train_labels.shape[1])

    # Addestramento del modello
    history = train_model(model, train_images, train_labels, val_images, val_labels, epochs, batch_size)

    # Salvataggio del modello addestrato
    model.save(model_save_path)
    print("Modello salvato con successo!")

    # Valutazione del modello
    evaluate_model(model, test_images, test_labels)

    # Grafici delle metriche
    plot_training_history(history)

if __name__ == "__main__":
    main()

