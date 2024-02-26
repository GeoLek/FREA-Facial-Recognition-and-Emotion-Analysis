import os
import cv2

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/jpg/FER-2013/'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')
classes = ['angry', 'happy', 'sad', 'surprise', 'fear', 'neutral', 'disgust']


def convert_jpg_to_png(data_dir):
    for emotion in classes:
        class_dir = os.path.join(data_dir, emotion)
        # Check if the class directory exists to avoid errors
        if not os.path.exists(class_dir):
            print(f"Directory does not exist: {class_dir}")
            continue

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith('.jpg'):
                img_path = os.path.join(class_dir, img_name)
                image = cv2.imread(img_path)

                # Define the output path with the same file name but with .png extension
                output_path = os.path.join(class_dir, os.path.splitext(img_name)[0] + '.png')

                # Save the image in PNG format
                cv2.imwrite(output_path, image)

                # Optionally, remove the original JPG file
                os.remove(img_path)


# Convert JPG images to PNG for both train and test datasets
convert_jpg_to_png(train_dir)
convert_jpg_to_png(test_dir)

print("Conversion from JPG to PNG completed.")
