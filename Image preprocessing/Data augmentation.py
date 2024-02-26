import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
import numpy as np

base_path = '/Work-data/'
output_path = '/Work-data/augmented'

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
target_dir = os.path.join(base_path, 'disgust')  # Example for 'contempt'
output_dir = os.path.join(output_path, 'disgust')  # Output directory for 'contempt'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# How many augmented images you want to generate per original image
num_augmented_images = 3

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
            if i >= num_augmented_images:
                break  # Stop after generating the desired number of augmented images

# This example shows the process for one directory.
# You would need to repeat this for each category, potentially adjusting the augmentation strategy.
