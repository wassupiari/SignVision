import opendatasets as od

# Link del dataset su Kaggle
dataset_url = "https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"

# Scaricare il dataset
od.download(dataset_url)

print("Download completato!")
