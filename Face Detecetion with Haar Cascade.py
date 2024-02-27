import cv2
import os

def preprocess_dataset(base_path, output_base_path, frontal_cascade_path, profile_cascade_path):
    frontal_face_cascade = cv2.CascadeClassifier(frontal_cascade_path)
    profile_face_cascade = cv2.CascadeClassifier(profile_cascade_path)

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

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = frontal_face_cascade.detectMultiScale(gray, 1.01, 0) #The lower the values, the more sensitive the model
                if len(faces) == 0:
                    faces = profile_face_cascade.detectMultiScale(gray, 1.01, 0)

                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (224, 224))
                    output_filename = f"{expression}_{filename}"
                    cv2.imwrite(os.path.join(output_dir, output_filename), face_resized)
                    break  # Assuming we only process the first detected face for each image


# Example usage
base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data'
output_base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Datasets/Work-data/Face-Detection'
frontal_cascade_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/haarcascades/haarcascade_frontalface_default.xml'  # Update this to your custom XML file's path
profile_cascade_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/haarcascades/haarcascade_profileface.xml'  # Update this to your custom XML file's path

# Preprocess the dataset
preprocess_dataset(base_path, output_base_path, frontal_cascade_path, profile_cascade_path)
