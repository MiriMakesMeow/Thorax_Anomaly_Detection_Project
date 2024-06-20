import glob
import numpy as np
from Models.cnn import create_cnn_model, model_fit
from keras.models import save_model, load_model
from keras.preprocessing.image import ImageDataGenerator
from preprocessing import create_directory_structure

# Daten für CNN ImageDataGenerator sortieren
image_paths = glob.glob("./Thorax/images/*.png")
labels_path = "./Thorax/labels.csv"
output_dir = "./Thorax/sorted_images"
create_directory_structure(image_paths, labels_path, output_dir, max_images_per_class=10000)  # 1000 Bilder pro Klasse

# ImageDataGenerator für Datenaugmentation und Normalisierung erstellen
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% der Daten für Validierung
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    './Thorax/sorted_images',
    target_size=(256, 256),
    batch_size=32,
    color_mode="grayscale",
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    './Thorax/sorted_images',
    target_size=(256, 256),
    batch_size=32,
    color_mode="grayscale",
    class_mode='binary',
    subset='validation'
)

epochs = 1000
batch_size = 16
input_shape = (256, 256, 1)

model = create_cnn_model(input_shape)

model_fit(model, train_generator, validation_generator, epochs, batch_size)

save_model(model, "cnn.keras")
