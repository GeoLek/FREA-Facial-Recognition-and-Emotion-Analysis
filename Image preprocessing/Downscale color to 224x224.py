import os
import cv2

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/Young AffectNet HQ'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')
classes = ['anger', 'happy', 'sad', 'surprise', 'fear', 'neutral', 'disgust', 'contempt']


def downscale_image(image):
    # Downscale the image to 224x224
    downscaled_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return downscaled_image


def process_images(data_dir, output_dir):
    for emotion in classes:
        class_dir = os.path.join(data_dir, emotion)
        output_class_dir = os.path.join(output_dir, emotion)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            # Read the image in color
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # Downscale the image
            image_downscaled = downscale_image(image)

            # Save the processed image
            output_path = os.path.join(output_class_dir, img_name)
            cv2.imwrite(output_path, image_downscaled)


# Define output directories for processed datasets
output_train_dir = os.path.join(base_path, 'processed_train2')
output_test_dir = os.path.join(base_path, 'processed_test2')

# Process the datasets
process_images(train_dir, output_train_dir)
process_images(test_dir, output_test_dir)
