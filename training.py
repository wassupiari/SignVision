
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model, train_generator, val_generator, epochs=30, batch_size=32):
    """Addestra il modello CNN con Data Augmentation e restituisce la history."""

    # Callback per prevenire overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Addestramento del modello con i generatori
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, reduce_lr]  # Aggiunto early stopping e riduzione LR
    )

    return history
