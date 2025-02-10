import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import classes

def evaluate_model(model, test_images, test_labels):
    """Valuta il modello e genera il report di classificazione e la matrice di confusione con i nomi delle classi."""
    y_pred_probs = model.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    class_labels = [classes[i] for i in range(len(classes))]

    print("\n\U0001F4CA Report di Classificazione:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 12))  # Ingrandisce la matrice di confusione
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predetto", fontsize=14)
    plt.ylabel("Reale", fontsize=14)
    plt.title("Matrice di Confusione", fontsize=16)
    plt.xticks(rotation=90, fontsize=12)  # Ruota e ingrandisce le etichette
    plt.yticks(rotation=0, fontsize=12)
    plt.show()
