from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def create_model(input_shape=(30, 30, 3), num_classes=43):
    """Definisce e restituisce il modello CNN con l'architettura aggiornata e Batch Normalization."""
    model = Sequential()

    # Primo blocco di convoluzione
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())  # <-- Batch Normalization
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.BatchNormalization())  # <-- Batch Normalization
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.15))

    # Secondo blocco di convoluzione
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.BatchNormalization())  # <-- Batch Normalization
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))

    # Parte fully connected
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())  # <-- Batch Normalization
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compilazione del modello
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_data_augmentation():
    """Restituisce un generatore di immagini con Data Augmentation."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
