import os
import cv2
from facenet_pytorch import MTCNN
import torch


def preprocess_dataset_mtcnn(base_path, output_base_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device, image_size=224)

    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for expression in classes:
        input_dir = os.path.join(base_path, expression)
        output_dir = os.path.join(output_base_path, expression)
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(input_dir, filename)
                img = cv2.imread(file_path)

                if img is None:
                    print(f"Warning: '{file_path}' could not be loaded and will be skipped.")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                boxes, _ = mtcnn.detect(img_rgb)
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        # Clamp the coordinates to the image dimensions
                        x1, y1 = max(x1, 0), max(y1, 0)
                        x2, y2 = min(x2, img_rgb.shape[1]), min(y2, img_rgb.shape[0])

                        # Check if the adjusted box is valid
                        if (x2 - x1) > 0 and (y2 - y1) > 0:
                            face = img_rgb[y1:y2, x1:x2]
                            if face.size > 0:
                                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                                output_filename = f"{filename.rsplit('.', 1)[0]}_{i}.png"
                                cv2.imwrite(os.path.join(output_dir, output_filename), face_bgr)
                            else:
                                print(f"Warning: Adjusted face is empty for '{file_path}'. Skipping.")
                        else:
                            print(f"Warning: Adjusted face is out of bounds for '{file_path}'. Skipping.")


# Example usage
base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data'
output_base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Datasets/Work-data/Face-Detection'

# Directly process the emotion class folders
preprocess_dataset_mtcnn(base_path, output_base_path)
