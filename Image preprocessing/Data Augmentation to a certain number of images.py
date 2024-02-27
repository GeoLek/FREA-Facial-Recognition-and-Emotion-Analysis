import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/'
output_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/augmented'

# Define your augmentation strategy
augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Directory to augment
target_dir = os.path.join(base_path, 'fear')  # Adjust to 'anger'
output_dir = os.path.join(output_path, 'fear')  # Output directory for 'anger'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Count current images
current_image_count = len([name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, name))])
goal_image_count = 17500
required_augmentations = goal_image_count - current_image_count

# Determine how many augmentations are needed per image, rounding up to ensure we reach the goal
import math
num_augmented_images_per_original = math.ceil(required_augmentations / current_image_count)

# Process images in the target directory
for filename in os.listdir(target_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other file types if needed
        image_path = os.path.join(target_dir, filename)
        image = load_img(image_path)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Generate augmented images
        i = 0
        for batch in augmentation.flow(image, batch_size=1,
                                        save_to_dir=output_dir,
                                        save_prefix='aug',
                                        save_format='png'):
            i += 1
            if i >= num_augmented_images_per_original:
                break  # Stop after generating the required number of augmented images

# Note: This may slightly overshoot the goal of 17,500 images due to rounding up.
