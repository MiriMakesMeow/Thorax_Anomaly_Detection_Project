from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import callbacks
import matplotlib.pyplot as plt
import numpy as np

def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("./loss_history.png")
    plt.close()  

# Anomalieerkennung auf Testdaten
def compute_reconstruction_error(original, reconstruction):
    return np.mean(np.square(original - reconstruction), axis=(1, 2, 3))

def detect_anomalies(model, image_batch, threshold):
    reconstructed_images = model.predict(image_batch)
    reconstruction_errors = compute_reconstruction_error(image_batch, reconstructed_images)
    return reconstruction_errors > threshold

def model_fit(model, X_train, batch_size, epochs, X_val):
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, X_val), callbacks=[early_stopping])
    plot_training_history(history)

def autoencoder_split(no_findings, findings):
    # Daten aufteilen
    indices_nf = np.arange(len(no_findings))

    # shuffling indices 
    np.random.shuffle(indices_nf)

    # slice data 90% training, 10% valid
    train_split = int(0.9 * len(no_findings))

    train_indices = indices_nf[:train_split]
    val_indices = indices_nf[train_split:]

    # extract data
    X_train = no_findings[train_indices]
    X_val = no_findings[val_indices]
    X_test = findings

    return X_train, X_val, X_test
