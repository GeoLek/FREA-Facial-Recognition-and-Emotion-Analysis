import cv2
import os
import numpy as np

def preprocess_dataset(base_path, output_base_path):
    # Load the pre-trained model and its configuration
    model_path = "/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    classes = ['anger', 'happy', 'sad', 'surprise', 'fear', 'neutral', 'disgust', 'contempt']

    for expression in classes:
        input_dir = os.path.join(base_path, expression)
        output_dir = os.path.join(output_base_path, expression)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(input_dir, filename)
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Failed to load image {file_path}. Skipping...")
                    continue

                (h, w) = img.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                net.setInput(blob)
                detections = net.forward()

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Confidence threshold
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        face = img[startY:endY, startX:endX]
                        face_resized = cv2.resize(face, (224, 224))
                        output_filename = f"{expression}_{filename}"
                        cv2.imwrite(os.path.join(output_dir, output_filename), face_resized)
                        break  # Assuming we only process the first detected face for each image

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data'
output_base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Datasets/Work-data/Face-Detection'

# Preprocess the dataset
preprocess_dataset(base_path, output_base_path)
