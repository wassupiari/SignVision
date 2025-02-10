import data_preprocessing
from model import create_model
from training import train_model
from evaluation import evaluate_model
from plot_metrics import plot_training_history
import os

# Configurazione
dir = "../SignVision/gtsrb-german-traffic-sign"

test_labels_path = "../SignVision/gtsrb-german-traffic-sign/Test.csv"
train_labels_path = "../SignVision/gtsrb-german-traffic-sign/Train.csv"
img_size = (30, 30)
epochs = 5
batch_size = 128
model_save_path = "models/5_epoche/model.h5"

def main():
    # Caricamento e preprocessing del dataset
    train_images, train_labels = data_preprocessing.load_data(train_labels_path, dir, img_size)
    train_images, val_images, train_labels, val_labels = data_preprocessing.split_data(train_images, train_labels)

    # Creazione dei generatori
    train_generator, val_generator = data_preprocessing.get_data_generators(train_images, train_labels, val_images, val_labels, batch_size)

    # Creazione del modello
    model = create_model(input_shape=(img_size[0], img_size[1], 3), num_classes=train_labels.shape[1])

    # Addestramento del modello
    history = train_model(model, train_generator, val_generator, epochs, batch_size)

    # Salvataggio del modello addestrato
    model.save(model_save_path)
    print("Modello salvato con successo!")

    # Caricamento e preprocessing del test set
    test_images, test_labels = data_preprocessing.load_data(test_labels_path, dir, img_size)
    evaluate_model(model, test_images, test_labels)


    # Grafici delle metriche
    plot_training_history(history, model, test_images, test_labels)

if __name__ == "__main__":
    main()
