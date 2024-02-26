import os
import cv2

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/Natural Human Face Images for Emotion Recognition'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')
classes = ['anger', 'happy', 'sad', 'surprise', 'fear', 'neutral', 'disgust', 'contempt']



def grayscale_to_color(image):
    # Convert the grayscale image to color by duplicating the grayscale channel
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return color_image


def process_images(data_dir, output_dir):
    for emotion in classes:
        class_dir = os.path.join(data_dir, emotion)
        output_class_dir = os.path.join(output_dir, emotion)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            # Read the image in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Convert the grayscale image to color
            image_color = grayscale_to_color(image)

            # Save the processed image
            output_path = os.path.join(output_class_dir, img_name)
            cv2.imwrite(output_path, image_color)


# Define output directories for processed datasets
output_train_dir = os.path.join(base_path, 'processed_train')
output_test_dir = os.path.join(base_path, 'processed_test')

# Process the datasets
process_images(train_dir, output_train_dir)
process_images(test_dir, output_test_dir)
