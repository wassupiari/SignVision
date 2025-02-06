import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import classes


def evaluate_model(model, test_images, test_labels, class_labels=None):
    """Valuta il modello e genera il report di classificazione e la matrice di confusione con le classi nominate."""
    y_pred_probs = model.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    print("\n\U0001F4CA Report di Classificazione:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.show()