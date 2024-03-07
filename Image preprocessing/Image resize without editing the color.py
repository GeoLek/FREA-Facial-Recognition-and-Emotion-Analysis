import os
import cv2


def resize_image_in_directory(input_dir, output_dir, size=(224, 224)):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # Check if it's a file and an image
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg')):
            img = cv2.imread(file_path)

            # Skip if the image cannot be loaded
            if img is None:
                print(f"Failed to load image {file_path}. Skipping...")
                continue

            # Resize and save the image
            resized_img = cv2.resize(img, size)
            output_file_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_file_path, resized_img)
            print(f"Resized and saved {output_file_path}")


def resize_images_for_classes(base_dir, output_base_path, classes):
    for class_name in classes:
        input_class_dir = os.path.join(base_dir, class_name)
        output_class_dir = os.path.join(output_base_path, class_name)

        print(f"Processing class: {class_name}")
        resize_image_in_directory(input_class_dir, output_class_dir)

# Directories
base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/Face-DetectionMTCNN/train'
output_base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/Face-DetectionMTCNN/resized'

classes = ['anger', 'happy', 'sad', 'surprise', 'fear', 'neutral', 'disgust', 'contempt']

# Call the function to resize images for each class
resize_images_for_classes(base_path, output_base_path, classes)

