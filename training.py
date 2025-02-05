from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    return history
