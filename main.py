import glob
import numpy as np
from import_and_processing import load_images_and_labels, load_and_preprocess, load_saved_images
from preprocessing import preprocessing
from data_slicing import shuffle_and_split
from NeuralNetwork.network import NetworkMatrix
from train import train
from NeuralNetwork.evaluation import evaluate
from autoencoder import autoencoder
from keras import callbacks

# import images
# image_paths = glob.glob("./Thorax/images/*.png")
# num_images = len(image_paths)
# labels_path = "./Thorax/labels.csv"

# labeled_images = load_images_and_labels(image_paths, labels_path)
# processed_images = preprocessing(labeled_images)

# images, labels, valid_index = load_and_preprocess(image_paths, labels_path, num_images, use_half=True)

images, labels, valid_index = load_saved_images()

X_train, y_train, X_val, y_val, X_test, y_test = shuffle_and_split(images, labels, valid_index)

# initialize network
#network = NetworkMatrix(number_inputs=2, number_hidden=10, number_outputs=1)

# initialize autoencoder
model = autoencoder

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train
epochs = 1000 
learning_rate = 0.3

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, batch_size = 256, epochs = epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])

#train(network, X_train, y_train, epochs, learning_rate)

# evaluate

#evaluate(network, X_val, y_val)
