import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_score



# Funzione per tracciare i grafici di addestramento
def plot_training_history(history, accuracy, recall, precision, specificity, y_test, y_probs, output_dir):
    """
    Plotta la loss, l'accuracy, la precision, il recall e la specificity del modello durante l'addestramento.
    """
    plt.figure(figsize=(15, 10))

    # Plot della Loss
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    # Plot dell'Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    # Grafico per accuracy, recall, precision e specificity
    metrics = ['Accuracy', 'Recall', 'Precision', 'Specificity']
    values = [accuracy, recall, precision, specificity]

    plt.subplot(2, 3, 3)
    plt.bar(metrics, [value * 100 for value in values], color=['blue', 'green', 'orange', 'purple'])
    plt.title('Comparison of Accuracy, Recall, Precision and Specificity')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)

    for i, v in enumerate(values):
        plt.text(i, v * 100 + 2, f"{v * 100:.2f}%", ha='center', color='black', fontsize=12)

    # Plot della Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.subplot(2, 3, 4)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # linea diagonale
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Salvataggio dei grafici
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_metrics.png")
    plt.show()

def compute_and_save_metrics(y_true, y_pred_binary, output_dir):
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_true, y_pred_binary)

    # Estrai i valori dalla matrice di confusione
    tn, fp, fn, tp = cm.ravel()

    # Calcola precision, recall, accuracy e specificity3
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0

    # Salva la matrice di confusione
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    return precision, recall, accuracy, specificity