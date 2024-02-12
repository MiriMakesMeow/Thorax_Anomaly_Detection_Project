import cv2
import numpy as np

def preprocessing(labeled_images, target_size=(1024, 1024)):
    processed_images = []

    for image, label in labeled_images:
        resized = cv2.resize(image, target_size)
        normalized = resized / 255
        processed_images.append(normalized, label)

    print('Preprocessed Images')
    
    return processed_images
