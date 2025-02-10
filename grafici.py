# Aggiornamento del codice per includere etichette delle classi nei grafici.

import os
import matplotlib.pyplot as plt
import pandas as pd

# Dizionario delle classi
classes = {
    0: 'Limite 20km/h', 1: 'Limite 30km/h', 2: 'Limite 50km/h', 3: 'Limite 60km/h', 4: 'Limite 70km/h', 5: 'Limite 80km/h',
    6: 'Fine limite 80km/h', 7: 'Limite 100km/h', 8: 'Limite 120km/h', 9: 'Divieto di sorpasso',
    10: 'Divieto sorpasso >3.5t', 11: 'Precedenza incrocio', 12: 'Strada prioritaria', 13: 'Dare precedenza', 14: 'Stop',
    15: 'Divieto di transito', 16: 'Divieto >3.5t', 17: 'Divieto di accesso', 18: 'Pericolo generico',
    19: 'Curva sinistra', 20: 'Curva destra', 21: 'Doppia curva', 22: 'Strada dissestata', 23: 'Strada sdrucciolevole',
    24: 'Strettoia destra', 25: 'Lavori in corso', 26: 'Semaforo', 27: 'Attraversamento pedoni',
    28: 'Attraversamento bambini', 29: 'Attraversamento ciclisti', 30: 'Ghiaccio/neve', 31: 'Animali selvatici',
    32: 'Fine limite e sorpasso', 33: 'Svolta destra', 34: 'Svolta sinistra', 35: 'Dritto', 36: 'Dritto o destra',
    37: 'Dritto o sinistra', 38: 'Tenere destra', 39: 'Tenere sinistra', 40: 'Rotatoria', 41: 'Fine divieto sorpasso',
    42: 'Fine divieto >3.5t'
}

# Percorsi dei file CSV
train_csv_path = 'gtsrb-german-traffic-sign/Train.csv'  # Percorso del CSV del training set
test_csv_path = 'gtsrb-german-traffic-sign/Test.csv'    # Percorso del CSV del test set

# Cartella di salvataggio dei grafici
output_dir = 'grafici'
os.makedirs(output_dir, exist_ok=True)

# Caricamento dei dataset
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

# Contare il numero di immagini per ogni classe
train_class_counts = train_data['ClassId'].value_counts().sort_index()
test_class_counts = test_data['ClassId'].value_counts().sort_index()

# Grafico 1: Distribuzione delle immagini per classe con etichette
plt.figure(figsize=(15, 8))
plt.bar([classes[i] for i in train_class_counts.index], train_class_counts.values, color='skyblue', label='Training')
plt.bar([classes[i] for i in test_class_counts.index], test_class_counts.values, color='orange', label='Test', alpha=0.7)
plt.title('Distribuzione delle immagini per classe', fontsize=16)
plt.xlabel('Classe', fontsize=14)
plt.ylabel('Numero di immagini', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.legend(fontsize=12)
plt.tight_layout()
grafico1_path = os.path.join(output_dir, 'distribuzione_per_classe_con_etichette.png')
plt.savefig(grafico1_path)
plt.close()

# Grafico 2: Confronto tra il numero di immagini nel set di training e test
plt.figure(figsize=(8, 6))
set_sizes = [len(train_data), len(test_data)]
set_labels = ['Training', 'Test']
colors = ['blue', 'orange']

plt.bar(set_labels, set_sizes, color=colors)
for i, size in enumerate(set_sizes):
    plt.text(i, size + 500, f'{size:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Confronto tra set di training e test', fontsize=16)
plt.ylabel('Numero di immagini', fontsize=14)
plt.tight_layout()
grafico2_path = os.path.join(output_dir, 'confronto_train_test.png')
plt.savefig(grafico2_path)
plt.close()

grafico1_path, grafico2_path
