import os
import cv2

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data'
directories = ['train', 'test', 'validation']
classes = ['anger', 'happy', 'sad', 'surprise', 'fear', 'neutral', 'disgust', 'contempt']

for dir in directories:
    dir_path = os.path.join(base_path, dir)
    for cls in classes:
        class_path = os.path.join(dir_path, cls)
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            # Read the image in color
            img = cv2.imread(file_path)
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Convert the grayscale image back to a 3 channel image
            gray_img_3channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            # Save the grayscale image back to the same location
            cv2.imwrite(file_path, gray_img_3channel)

print("All images have been converted to grayscale.")
