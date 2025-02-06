from model import get_data_augmentation


def train_model(model, train_images, train_labels, val_images, val_labels, epochs=30, batch_size=32):
    """Addestra il modello CNN con Data Augmentation e restituisce la history."""

    # Otteniamo il generatore di immagini con Data Augmentation
    train_datagen = get_data_augmentation()

    # Creiamo il generatore di immagini per il set di training
    train_generator = train_datagen.flow(
        train_images, train_labels, batch_size=batch_size
    )

    # Addestramento del modello con il generatore
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(val_images, val_labels)
    )

    return history
