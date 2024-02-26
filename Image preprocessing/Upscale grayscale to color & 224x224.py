import os
import cv2
from PIL import Image
import numpy as np

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/CK+ (Extended Cohn-Kanade Dataset)/'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')
classes = ['angry', 'happy', 'sadness', 'surprise', 'fear', 'contempt', 'disgust']


def upscale_and_colorize(image):
    # Upscale the image to the target resolution
    upscaled_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    # Convert the upscaled grayscale image to color (RGB) by replicating the grayscale channel
    color_image = cv2.cvtColor(upscaled_image, cv2.COLOR_GRAY2RGB)
    return color_image


def apply_super_resolution(image):
    # Placeholder for super-resolution process
    # You'll need to replace this with the actual implementation
    # For now, we'll return the image as is
    return image


def process_images(data_dir, output_dir):
    for emotion in classes:
        class_dir = os.path.join(data_dir, emotion)
        output_class_dir = os.path.join(output_dir, emotion)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # First, upscale and convert the image to color
            image_color = upscale_and_colorize(image)

            # Then, apply super-resolution to enhance the image quality
            image_sr = apply_super_resolution(image_color)

            # Save the processed image
            output_path = os.path.join(output_class_dir, img_name)
            cv2.imwrite(output_path, image_sr)


# Define output directories for processed datasets
output_train_dir = os.path.join(base_path, 'processed_train')
output_test_dir = os.path.join(base_path, 'processed_test')

# Process the datasets
process_images(train_dir, output_train_dir)
process_images(test_dir, output_test_dir)
