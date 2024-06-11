import glob
import numpy as np
from import_and_processing import load_images_and_labels, load_and_preprocess, load_saved_images
from data_slicing import shuffle_and_split
from autoencoder import autoencoder
from keras import callbacks

# Daten importieren und vorverarbeiten
image_paths = glob.glob("./Thorax/images/*.png")
num_images = len(image_paths)
labels_path = "./Thorax/labels.csv"

# Bilder und Labels laden und vorverarbeiten
no_findings, findings = load_and_preprocess(image_paths, labels_path, num_images, use_half=True)
# oder, wenn bereits gespeichert:
# no_findings, findings = load_saved_images()

# Daten aufteilen
# TODO: auf no findings anpassen X_train, y_train, X_val, y_val, X_test, y_test = shuffle_and_split(images, labels, valid_index)

# Autoencoder initialisieren
model = autoencoder

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training konfigurieren
epochs = 100
batch_size = 32

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, X_val), callbacks=[early_stopping])

# Anomalieerkennung auf Testdaten
def compute_reconstruction_error(original, reconstruction):
    return np.mean(np.square(original - reconstruction), axis=(1, 2, 3))

def detect_anomalies(image_batch, threshold):
    reconstructed_images = model.predict(image_batch)
    reconstruction_errors = compute_reconstruction_error(image_batch, reconstructed_images)
    return reconstruction_errors > threshold

# Anomalien erkennen
anomalies_detected = detect_anomalies(X_test, threshold=0.01)  # Beispielhafter Schwellenwert
print("Anomalies detected:", anomalies_detected)