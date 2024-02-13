import cv2
import os
import pandas as pd
import numpy as np

def load_and_preprocess(image_paths, labels_path, num_images, use_half = True):
    
    labels_df = pd.read_csv(labels_path)    # import labels

    # import as np
    if use_half:
        num_images = len(image_paths) // 2
        image_paths = image_paths[:num_images]
    image_width, image_height = 256, 256
    num_channels = 1 # gray
    images = np.zeros((num_images, image_height, image_width, num_channels), dtype=np.uint8)
    labels = np.zeros((num_images), dtype=int)
    valid_index = 0

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # check if image was loaded correctly
        if image is None:
            continue
        
        resized_image = cv2.resize(image, (256, 256))
        # extract file name
        file_name = os.path.basename(path)

        # extract label from dataframe and add to array
        label_row = labels_df[labels_df['Image Index'] == file_name]
        if label_row.empty:
            continue
        label = label_row["Finding Labels"].values[0]
        if "No Finding" in label:
            label = 0
        else:
            label = 1

        labels[valid_index] = label 

        # add image to array, incl reshaping
        images[valid_index] = resized_image.reshape(256, 256, num_channels)

        valid_index += 1
        if valid_index % 1000 == 0:
            print(f'Loaded Image No. {valid_index}')

    # cut arrays to correct num
    images = images[:valid_index]
    labels = labels[:valid_index]

    # normalize and save
    images = images.astype("float16")/ 255.0

    np.save("images.npy", images)
    np.save("labels.npy", labels)

    print(f'Loaded {valid_index} Images')

    return images, labels, valid_index

def load_saved_images():
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    valid_index = len(images)

    print(f'Loaded {valid_index} images')
    return images, labels, valid_index


def load_images_and_labels(image_paths, labels_path):
    labels_df = pd.read_csv(labels_path)    # import labels

    # import as list, no copy
    labeled_images = []

    for file_path in image_paths:
        image = cv2.imread(file_path)
        if image is None: # check if image was loaded correctly
            continue # skip

        file_name = os.path.basename(file_path) # extract file name

        label_row = labels_df.loc[labels_df['Image Index'] == file_name] # match labels with filename

        if label_row.empty: 
            continue # skip if label row didn't have label
        label = label_row['Finding Labels'].values[0] # extract labels

        # check if label is finding
        if "No Finding" in label:
            label = 0   # "No Finding"
        else:
            label = 1   # "Finding"
        labeled_images.append((image, label))

    print('Loaded Images')

    return labeled_images
