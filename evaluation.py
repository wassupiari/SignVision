import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_images, test_labels):
    """Valuta il modello e stampa il report di classificazione e la matrice di confusione."""
    y_pred_probs = model.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    print("\n\U0001F4CA Report di Classificazione:\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.show()