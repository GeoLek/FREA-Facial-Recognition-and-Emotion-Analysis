#Rename your images. For example from img023394.png to happy001.png, happy002.png and so on

import os


def rename_files(folder_path, prefix="surprise"): #prefix of the word you want to be renamed to
    # Initialize a counter for the new file names
    index = 1

    # List all files in the directory
    files = os.listdir(folder_path)
    # Filter out directories, keeping only files
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    # Sort files for consistent ordering (optional)
    files.sort()

    # Loop through each file and rename it
    for file_name in files:
        # Define the new file name, initially with the current index
        new_name = f"{prefix}{index:03}.png"
        # Define the full path for the new file
        new_file = os.path.join(folder_path, new_name)

        # Check if a file with the new name already exists, increment index until it doesn't
        while os.path.exists(new_file):
            index += 1
            new_name = f"{prefix}{index:03}.png"
            new_file = os.path.join(folder_path, new_name)

        # Define the full old file path
        old_file = os.path.join(folder_path, file_name)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed '{file_name}' to '{new_name}'")

        # Increment the index for the next iteration
        index += 1
# Example usage
folder_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/surprise'  # Change this to your folder's path
rename_files(folder_path)
