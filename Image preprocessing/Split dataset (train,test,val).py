import os
import random
import shutil

# Define your dataset directory and the output directories for splits
dataset_dir = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/surprise'
output_dir = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/Split'

# Directories for train, validation, and test splits
train_dir = os.path.join(output_dir, 'train')
validation_dir = os.path.join(output_dir, 'validation')
test_dir = os.path.join(output_dir, 'test')

# Make sure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read all file names directly from the dataset directory
file_list = [f for f in os.listdir(dataset_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Apply a really more random shuffle
random.shuffle(file_list)

# Define split ratios
train_ratio = 0.7
validation_ratio = 0.15
# The remaining for test_ratio

# Calculate split indices
train_split = int(train_ratio * len(file_list))
validation_split = train_split + int(validation_ratio * len(file_list))

# Move files to the respective directories
for i, file_name in enumerate(file_list):
    src_path = os.path.join(dataset_dir, file_name)

    if i < train_split:
        dst_path = os.path.join(train_dir, file_name)
    elif i < validation_split:
        dst_path = os.path.join(validation_dir, file_name)
    else:
        dst_path = os.path.join(test_dir, file_name)

    shutil.move(src_path, dst_path)

print(
    f"Dataset split: {train_ratio * 100}% train, {validation_ratio * 100}% validation, {len(file_list) - validation_split}% test.")



