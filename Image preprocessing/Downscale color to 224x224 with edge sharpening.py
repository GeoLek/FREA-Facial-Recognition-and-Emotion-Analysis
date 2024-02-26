import os
import cv2
import numpy as np

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/Young AffectNet HQ'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')
classes = ['anger', 'happy', 'sad', 'surprise', 'fear', 'neutral', 'disgust', 'contempt']


def apply_sharpening(image):
    # Define a kernel for sharpening
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # Apply the kernel to the image
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image


def downscale_and_preprocess(image):
    # Downscale the image to 224x224
    downscaled_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Apply sharpening to preserve edge details
    processed_image = apply_sharpening(downscaled_image)
    return processed_image


def process_images(data_dir, output_dir):
    for emotion in classes:
        class_dir = os.path.join(data_dir, emotion)
        output_class_dir = os.path.join(output_dir, emotion)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            # Read the image in color
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # Downscale the image and apply preprocessing to preserve edge details
            image_processed = downscale_and_preprocess(image)

            # Save the processed image
            output_path = os.path.join(output_class_dir, img_name)
            cv2.imwrite(output_path, image_processed)


# Define output directories for processed datasets
output_train_dir = os.path.join(base_path, 'processed_train')
output_test_dir = os.path.join(base_path, 'processed_test')

# Process the datasets
process_images(train_dir, output_train_dir)
process_images(test_dir, output_test_dir)