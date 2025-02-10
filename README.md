
# SignVision 👁️: Riconoscimento di Segnali Stradali con CNN

## 📌 Descrizione

---
SignVision è un modello basato su Reti Neurali Convoluzionali ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) per il riconoscimento automatico dei segnali stradali. Il progetto è stato sviluppato nell'ambito del corso di Fondamenti di Intelligenza Artificiale per la classificazione di segnali stradali utilizzando il dataset [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (German Traffic Sign Recognition Benchmark).

Il progetto include:

- Pre-elaborazione dei dati (ridimensionamento, normalizzazione, Data Augmentation)

- Addestramento di una CNN con TensorFlow/Keras

- Valutazione delle prestazioni tramite grafici di accuratezza e matrice di confusione

- Interfaccia Web con Flask per testare il modello
## 🚀 Funzionalità

---

## 📂 Struttura del Progetto
📁gui/                    #interfaccia web    
    ├── static/           # Contiene CSS, immagini e file statici<br>
    ├── templates/        # File HTML per Flask<br>
    ├── app.py            # Applicazione Flask<br>
📁 grafici/               # Contiene i grafici dei risultati<br>
📁 models/                # Contiene il modello addestrato (.h5)<br>
📁 dataset/               # Dataset utilizzato per l'addestramento<br>
📁 scripts/               # Script Python per addestramento e valutazione<br>
    ├── data_preprocessing.py  # Pre-elaborazione del dataset<br>
    ├── training.py            # Addestramento del modello<br>
    ├── evaluation.py          # Valutazione e metriche<br>
    ├── model.py               # Definizione della CNN<br>
README.md                # Documentazione del progetto<br>
requirements.txt         # Librerie richieste
---
## 📥 Pre-requisiti

---

---

### **Autori**: [Iari Normanno](https://github.com/wassupiari), [Marco Acierno](https://github.com/m4rc00000)<br>
### **Università**: Università degli Studi di Salerno<br>
### **Anno Accademico**: 2024/2025


