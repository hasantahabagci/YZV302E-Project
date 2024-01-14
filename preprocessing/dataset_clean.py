import os
import random

def delete_random_images(folder_path, number_to_keep):
    # Get all file names in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Filter out non-image files if necessary (optional)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Randomly select files to keep
    files_to_keep = random.sample(image_files, min(number_to_keep, len(image_files)))

    # Delete all other files
    count = 0
    for file in image_files:
        if file not in files_to_keep:
            os.remove(os.path.join(folder_path, file))
            #print(f"Deleted {file}")
            count += 1
    print(f"Deleted {count} files")

folder_path = '../American'  
number_to_keep = 70  

subfolders = os.listdir(folder_path)
for subfolder in subfolders:
    if subfolder != '.DS_Store':
        subfolder_path = os.path.join(folder_path, subfolder)
        delete_random_images(subfolder_path, number_to_keep)
