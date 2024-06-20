import glob
import numpy as np
from import_and_processing import load_and_preprocess, load_saved_images
from sklearn.model_selection import train_test_split
from Models.autoencoder import build_autoencoder, model_fit, detect_anomalies, autoencoder_split
from keras.models import save_model, load_model

# Daten importieren und vorverarbeiten
image_paths = glob.glob("./Thorax/images/*.png") 
num_images = len(image_paths)
labels_path = "./Thorax/labels.csv"

# Bilder und Labels laden und vorverarbeiten
no_findings, findings = load_and_preprocess(image_paths, labels_path, num_images, use_half=True)
# oder, wenn bereits gespeichert:
# no_findings, findings = load_saved_images()

input_shape = (256, 256, 1)

# Training konfigurieren
epochs = 100
batch_size = 16

# Autoencoder
X_train, X_val, X_test = autoencoder_split(no_findings, findings)

# Autoencoder initialisieren
model = build_autoencoder(input_shape)

model_fit(model, X_train, batch_size, epochs, X_val)

save_model(model, "autoencoder.keras")
# model = model.keras.load_model("autoencoder.keras")

# Anomalien erkennen
anomalies_detected = detect_anomalies(X_test, threshold=0.01)  # Beispielhafter Schwellenwert
print("Anomalies detected:", anomalies_detected)
