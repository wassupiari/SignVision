import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
import tensorflow as tf

def plot_training_history(history, model, test_images, test_labels):
    """Genera e salva i grafici relativi all'addestramento e alla valutazione del modello."""

    # Creazione della cartella di output per i grafici
    output_dir = "grafici"
    os.makedirs(output_dir, exist_ok=True)

    # Estrazione delle metriche dal training history
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    # Grafico Accuratezza
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuratezza durante l\'addestramento')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

    # Grafico Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Loss durante l\'addestramento')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Predizioni sul test set
    y_probs = model.predict(test_images)  # Probabilità per ogni classe
    y_pred = np.argmax(y_probs, axis=1)  # Classe predetta
    y_true = np.argmax(test_labels, axis=1)  # Classe reale

    # Generazione del report di classificazione
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = report['accuracy']
    recall = np.mean([report[str(i)]['recall'] for i in range(len(report) - 3)])
    precision = np.mean([report[str(i)]['precision'] for i in range(len(report) - 3)])

    # Grafico per Accuracy, Recall e Precision
    metrics = ['Accuracy', 'Recall', 'Precision']
    values = [accuracy, recall, precision]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, [value * 100 for value in values], color=['blue', 'green', 'orange'])
    plt.title('Confronto tra Accuracy, Recall e Precision')
    plt.ylabel('Percentuale')
    plt.ylim(0, 100)

    for i, v in enumerate(values):
        plt.text(i, v * 100 + 2, f"{v*100:.2f}%", ha='center', color='black', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_recall_precision_plot.png'))
    plt.close()

    # Calcolo della curva ROC e dell'AUC per una classe media
    y_test_ravel = test_labels.ravel()
    y_probs_ravel = y_probs.ravel()
    fpr, tpr, _ = roc_curve(y_test_ravel, y_probs_ravel)
    roc_auc = auc(fpr, tpr)

    # Grafico ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Linea diagonale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
