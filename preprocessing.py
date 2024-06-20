import cv2
import pandas as pd
import os

def preprocessing(labeled_images, target_size=(1024, 1024)):
    processed_images = []

    for image, label in labeled_images:
        resized = cv2.resize(image, target_size)
        normalized = resized / 255
        processed_images.append(normalized, label)

    print('Preprocessed Images')
    
    return processed_images

def create_directory_structure(image_paths, labels_path, output_dir, max_images_per_class=None):
    labels_df = pd.read_csv(labels_path)
    image_width, image_height = 256, 256

    no_finding_dir = os.path.join(output_dir, 'No_Finding')
    finding_dir = os.path.join(output_dir, 'Finding')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(no_finding_dir):
        os.makedirs(no_finding_dir)
    if not os.path.exists(finding_dir):
        os.makedirs(finding_dir)

    no_finding_count = 0
    finding_count = 0

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        resized_image = cv2.resize(image, (image_width, image_height))
        file_name = os.path.basename(path)
        label_row = labels_df[labels_df['Image Index'] == file_name]
        if label_row.empty:
            continue
        label = label_row["Finding Labels"].values[0]
        if "No Finding" in label:
            output_path = os.path.join(no_finding_dir, file_name)
            if max_images_per_class and no_finding_count >= max_images_per_class:
                continue
            no_finding_count += 1
        else:
            output_path = os.path.join(finding_dir, file_name)
            if max_images_per_class and finding_count >= max_images_per_class:
                continue
            finding_count += 1

        cv2.imwrite(output_path, resized_image)
        print(f"Processed {file_name}")

        if max_images_per_class and no_finding_count >= max_images_per_class and finding_count >= max_images_per_class:
            break
