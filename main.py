from data_preprocessing import load_data, split_data
from model import create_model
from training import train_model
from evaluation import evaluate_model
from plot_metrics import plot_training_history
import os

# Configurazione
train_data_path = "Train"
test_data_path = "Test"
test_labels_path = "Test.csv"
img_size = (30, 30)
epochs = 30
batch_size = 32
model_save_path = "models/traffic_signs_model.h5"

# Caricamento e preprocessing del dataset
train_images, train_labels = load_data(test_data_path, test_labels_path, img_size)
train_images, val_images, train_labels, val_labels = split_data(train_images, train_labels)

# Creazione del modello
model = create_model(input_shape=(img_size[0], img_size[1], 3), num_classes=train_labels.shape[1])

# Addestramento del modello
history = train_model(model, train_images, train_labels, val_images, val_labels, epochs, batch_size)

# Salvataggio del modello addestrato
model.save(model_save_path)
print("Modello salvato con successo!")

# Valutazione del modello
evaluate_model(model, val_images, val_labels)

# Grafici delle metriche
plot_training_history(history)
