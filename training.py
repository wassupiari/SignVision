import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import os


def train_model(model, train_generator, val_generator, num_classes, epochs=None, class_weights=None):
    """Allena il modello con gestione degli squilibri di classe."""

    # Se i pesi delle classi non sono forniti, calcolarli automaticamente
    if class_weights is None:
        print("[INFO] Calcolo automatico dei pesi delle classi...")
        y_train = np.argmax(train_generator.labels, axis=1)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
        class_weights = dict(enumerate(class_weights))  # Convertire in dizionario

    print("[INFO] Pesi assegnati alle classi:", class_weights)

    # Definizione dei callback per migliorare il training
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    # Allenamento del modello con pesi delle classi
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,  # Ora può ricevere i pesi sia dal main che calcolarli da solo
        callbacks=[early_stopping, reduce_lr]
    )

    return history
