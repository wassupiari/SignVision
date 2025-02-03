import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 🔹 Imposta i percorsi delle cartelle del dataset
dataset_path = "datasets/fer2013/train"
test_path = "datasets/fer2013/test"
output_path = "Grafici"  # Cartella dove salvare i grafici
os.makedirs(output_path, exist_ok=True)  # Crea la cartella se non esiste

def count_images_per_class(dataset_path):
    """Conta il numero di immagini per ogni classe in una cartella"""
    class_counts = {}
    total_images = 0  # Contatore totale immagini
    classes = os.listdir(dataset_path)
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            num_images = len(os.listdir(class_dir))
            class_counts[class_name] = num_images
            total_images += num_images
    return class_counts, total_images

# 🔹 Conta le immagini per training, test e totale
train_counts, train_total = count_images_per_class(dataset_path)
test_counts, test_total = count_images_per_class(test_path)

# 🔹 Calcola la somma delle immagini per ogni classe (Train + Test)
total_counts = {class_name: train_counts.get(class_name, 0) + test_counts.get(class_name, 0) for class_name in set(train_counts) | set(test_counts)}
all_total = sum(total_counts.values())

print(f"🔹 Totale immagini nel Training Set: {train_total}")
print(f"🔹 Totale immagini nel Test Set: {test_total}")
print(f"🔹 Totale immagini nel Dataset: {all_total}")

def plot_distribution(counts, title, filename):
    """Genera e salva il grafico della distribuzione delle classi con etichette sopra le barre"""
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.xticks(rotation=45)
    plt.xlabel("Classe")
    plt.ylabel("Numero di immagini")
    plt.title(title)

    # Aggiunge il numero di immagini sopra le barre
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', weight='bold')

    plt.savefig(os.path.join(output_path, filename))  # Salva il grafico
    plt.show()

# 🔹 Grafico distribuzione per Training Set
plot_distribution(train_counts, "Distribuzione delle immagini per classe (Train)", "distribuzione_classi_train.png")

# 🔹 Grafico distribuzione per Test Set
plot_distribution(test_counts, "Distribuzione delle immagini per classe (Test)", "distribuzione_classi_test.png")

# 🔹 Grafico distribuzione totale (Train + Test)
plot_distribution(total_counts, "Distribuzione totale delle immagini per classe (Train + Test)", "distribuzione_classi_totale.png")

# 🔹 Mostra e salva alcune immagini campione
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
classes = list(train_counts.keys())

for i, ax in enumerate(axes.flat):
    class_name = np.random.choice(classes)
    img_name = np.random.choice(os.listdir(os.path.join(dataset_path, class_name)))
    img_path = os.path.join(dataset_path, class_name, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converti da BGR a RGB per matplotlib
    ax.imshow(img)
    ax.set_title(class_name)
    ax.axis("off")

plt.tight_layout()

# 🔹 Salva il grafico delle immagini campione
plt.savefig(os.path.join(output_path, "esempio_immagini.png"))
plt.show()

print(f"📂 Tutti i grafici sono stati salvati nella cartella '{output_path}'.")
