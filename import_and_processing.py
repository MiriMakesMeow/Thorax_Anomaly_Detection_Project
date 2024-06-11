import cv2
import os
import pandas as pd
import numpy as np

def load_and_preprocess(image_paths, labels_path, num_images, use_half=True):
    labels_df = pd.read_csv(labels_path)  # Import labels

    if use_half:
        num_images = len(image_paths) // 2
        image_paths = image_paths[:num_images]
    image_width, image_height = 256, 256
    num_channels = 1  # gray

    no_finding_images = []
    finding_images = []

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        resized_image = cv2.resize(image, (256, 256))
        file_name = os.path.basename(path)

        label_row = labels_df[labels_df['Image Index'] == file_name]
        if label_row.empty:
            continue
        label = label_row["Finding Labels"].values[0]
        if "No Finding" in label:
            label = 0
            no_finding_images.append(resized_image.reshape(256, 256, num_channels))
        else:
            label = 1
            finding_images.append(resized_image.reshape(256, 256, num_channels))

    no_finding_images = np.array(no_finding_images).astype("float16") / 255.0
    finding_images = np.array(finding_images).astype("float16") / 255.0

    np.save("no_finding_images.npy", no_finding_images)
    np.save("finding_images.npy", finding_images)

    print(f'Loaded {len(no_finding_images)} no finding images')
    print(f'Loaded {len(finding_images)} finding images')

    return no_finding_images, finding_images

def load_saved_images():
    no_finding_images = np.load("no_finding_images.npy")
    finding_images = np.load("finding_images.npy")

    print(f'Loaded {len(no_finding_images)} no finding images')
    print(f'Loaded {len(finding_images)} finding images')
    return no_finding_images, finding_images

def load_images_and_labels(image_paths, labels_path):
    labels_df = pd.read_csv(labels_path)  # Import labels

    labeled_images = []

    for file_path in image_paths:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:  # Check if image was loaded correctly
            continue  # Skip

        file_name = os.path.basename(file_path)  # Extract file name

        label_row = labels_df.loc[labels_df['Image Index'] == file_name]  # Match labels with filename

        if label_row.empty:
            continue  # Skip if label row didn't have label
        label = label_row['Finding Labels'].values[0]  # Extract labels

        # Check if label is finding
        if "No Finding" in label:
            label = 0  # "No Finding"
        else:
            label = 1  # "Finding"
        labeled_images.append((image, label))

    print('Loaded Images')

    return labeled_images