def train_model(model, train_images, train_labels, val_images, val_labels, epochs=30, batch_size=32):
    """Addestra il modello CNN e restituisce la history."""
    history = model.fit(
        train_images, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_images, val_labels)
    )
    return history