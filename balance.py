import os
import shutil
import pandas as pd

# Percorsi
original_csv = "../MoodRipple/gtsrb-german-traffic-sign/Train.csv"
original_dir = "../MoodRipple/gtsrb-german-traffic-sign/"
filtered_csv = "../MoodRipple/gtsrb-german-traffic-sign-filtered/Train.csv"
filtered_dir = "../MoodRipple/gtsrb-german-traffic-sign-filtered/"

# Creare la cartella per il nuovo dataset
os.makedirs(filtered_dir, exist_ok=True)

# Leggere il CSV originale
data = pd.read_csv(original_csv)

# Filtrare le immagini con dimensioni tra 50x50 e 60x60
filtered_data = data[(data["Width"] >= 25) & (data["Width"] <= 50) &
                     (data["Height"] >= 25) & (data["Height"] <= 50)]

# Creare le cartelle per ogni categoria
for class_id in filtered_data["ClassId"].unique():
    os.makedirs(os.path.join(filtered_dir, str(class_id)), exist_ok=True)

# Copiare le immagini selezionate nella nuova cartella
new_data = []
for _, row in filtered_data.iterrows():
    old_path = os.path.join(original_dir, row["Path"])
    new_path = os.path.join(filtered_dir, str(row["ClassId"]), os.path.basename(row["Path"]))

    if os.path.exists(old_path):
        shutil.copy(old_path, new_path)
        new_data.append([os.path.join(str(row["ClassId"]), os.path.basename(row["Path"])), row["ClassId"]])

# Creare un nuovo CSV con i dati filtrati
filtered_df = pd.DataFrame(new_data, columns=["Path", "ClassId"])
filtered_df.to_csv(filtered_csv, index=False)

print(f"[INFO] Nuovo dataset filtrato salvato in {filtered_dir}")
print(f"[INFO] Nuovo CSV creato: {filtered_csv}")
