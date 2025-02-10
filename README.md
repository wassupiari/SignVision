# SignVision 👁️: Riconoscimento di Segnali Stradali con CNN

## 📌 Descrizione
SignVision è un modello basato su Reti Neurali Convoluzionali ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) per il riconoscimento automatico dei segnali stradali. Il progetto è stato sviluppato nell'ambito del corso di Fondamenti di Intelligenza Artificiale per la classificazione di segnali stradali utilizzando il dataset [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (German Traffic Sign Recognition Benchmark).

Il progetto include:

- Pre-elaborazione dei dati (ridimensionamento, normalizzazione, Data Augmentation)

- Addestramento di una CNN con TensorFlow/Keras

- Valutazione delle prestazioni tramite grafici di accuratezza e matrice di confusione

- Interfaccia Web con Flask per testare il modello
## 🚀 Funzionalità
Questo progetto offre diverse funzionalità chiave:
- **Caricamento e Pre-elaborazione delle Immagini:** Conversione delle immagini in un formato utilizzabile dal modello.
- **Classificazione Automatica:** Utilizzo di un modello CNN per predire il tipo di segnale stradale.
- **Visualizzazione delle Performance:** Analisi dettagliata tramite grafici e matrici di confusione.
- **Interfaccia Web Intuitiva:** Permette di caricare immagini e ottenere predizioni direttamente dal browser.

## 📂 **Struttura del Repository**
```
📁 gui/                   # Interfaccia Web
│   ├── static/           # Contiene CSS, immagini e file statici
│   ├── templates/        # File HTML per Flask
│   ├── app.py            # Applicazione Flask
│
📁 grafici/               # Contiene i grafici dei risultati
│
📁 models/                # Contiene il modello addestrato (.h5)
│
📁 dataset/               # Dataset utilizzato per l'addestramento
│
📁 scripts/               # Script Python per addestramento e valutazione
│   ├── data_preprocessing.py  # Pre-elaborazione del dataset
│   ├── training.py            # Addestramento del modello
│   ├── evaluation.py          # Valutazione e metriche
│   ├── model.py               # Definizione della CNN
│
README.md                # Documentazione del progetto
requirements.txt         # Librerie richieste
```

---

## 🚀 **Installazione e Utilizzo**

### **1️⃣ Installare le dipendenze**
Prima di eseguire il progetto, assicurati di installare le librerie necessarie:
```bash
pip install -r requirements.txt
```

### **2️⃣ Addestrare il modello**
Se vuoi addestrare il modello da zero, esegui:
```bash
python .\main.py
```
Il modello verrà salvato nella cartella `models/`.

### **3️⃣ Avviare l'Interfaccia Web**
Dopo aver addestrato il modello (o averne caricato uno preaddestrato), puoi testarlo tramite l'interfaccia web Flask:
```bash
python gui/app.py
```
Ora puoi accedere all'interfaccia da **http://127.0.0.1:5000/** e caricare un'immagine per la classificazione.

---

## 📊 **Risultati del Modello**
Il modello ha ottenuto un'accuratezza di **96%** sul test set.
- **Grafici di Accuratezza e Loss** suddivisi per epoche disponibili in `models/`
- **Matrice di Confusione** per analizzare le prestazioni sulle classi

---

## 🛠 **Tecnologie Utilizzate**
- **Python** 🐍
- **TensorFlow/Keras** 🔥
- **Flask** 🌐 (per l'interfaccia web)
- **Matplotlib & Seaborn** 📊 (per la visualizzazione dei dati)
- **OpenCV** 📷 (per la gestione delle immagini)

---

## 👨‍💻 **Autori**: [Iari Normanno](https://github.com/wassupiari), [Marco Acierno](https://github.com/m4rc00000)<br>
### **Università**: Università degli Studi di Salerno<br>
### **Anno Accademico**: 2024/2025