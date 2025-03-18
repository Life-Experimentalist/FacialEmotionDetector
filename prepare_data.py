import os
import shutil
import zipfile

import cv2
import kagglehub
import numpy as np

from utils import get_face_landmarks

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure the data directory exists and is populated
data_dir = os.path.join(script_dir, "data")
download_required = not os.path.exists(data_dir) or not os.listdir(data_dir)

if download_required:
    # Download the dataset from KaggleHub
    zip_path = kagglehub.dataset_download(
        "jonathanoheix/face-expression-recognition-dataset"
    )

    # Create a temporary directory for extraction
    temp_extract_dir = os.path.join(script_dir, "temp_extract")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir)

    # Extract contents of the zip file into temp directory
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # Create the final data directory structure
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Reorganize the directory structure to match the desired format
    src_train_dir = os.path.join(temp_extract_dir, "images", "train")
    src_validation_dir = os.path.join(temp_extract_dir, "images", "validation")

    target_train_dir = os.path.join(data_dir, "train")
    target_test_dir = os.path.join(data_dir, "test")

    # Create target directories if they don't exist
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)

    # Move all emotion folders from source to target
    for emotion in os.listdir(src_train_dir):
        src_emotion_path = os.path.join(src_train_dir, emotion)
        if os.path.isdir(src_emotion_path):
            target_emotion_path = os.path.join(target_train_dir, emotion)
            if os.path.exists(target_emotion_path):
                shutil.rmtree(target_emotion_path)
            shutil.copytree(src_emotion_path, target_emotion_path)

    for emotion in os.listdir(src_validation_dir):
        src_emotion_path = os.path.join(src_validation_dir, emotion)
        if os.path.isdir(src_emotion_path):
            target_emotion_path = os.path.join(target_test_dir, emotion)
            if os.path.exists(target_emotion_path):
                shutil.rmtree(target_emotion_path)
            shutil.copytree(src_emotion_path, target_emotion_path)

    # Clean up temporary directory
    shutil.rmtree(temp_extract_dir)

    print("Dataset extracted and reorganized to:", os.path.abspath(data_dir))

# Use the updated data structure: data/train/emotion_class
train_data_dir = os.path.join(data_dir, "train")

output = []
# Process each emotion directory
for emotion_indx, emotion in enumerate(sorted(os.listdir(train_data_dir))):
    emotion_dir = os.path.join(train_data_dir, emotion)
    if not os.path.isdir(emotion_dir):
        continue

    print(f"Processing {emotion} images...")
    for image_name in os.listdir(emotion_dir):
        image_path = os.path.join(emotion_dir, image_name)

        if not os.path.isfile(image_path):
            continue

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            face_landmarks = get_face_landmarks(image)

            if len(face_landmarks) >= 1404:
                face_landmarks.append(int(emotion_indx))
                output.append(face_landmarks)
            else:
                print(
                    f"Could not extract face landmarks for {image_path} and only got {len(face_landmarks)} landmarks"
                )
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if output:
    output_file = os.path.join(script_dir, "data.txt")
    np.savetxt(output_file, np.asarray(output))
    print(f"Processed {len(output)} images successfully")
else:
    print("No valid face landmarks were extracted. Check the data directory structure.")
    print()
