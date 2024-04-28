import os
import shutil
import random

# Define paths
dataset_dir = 'posters/'
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

# Create directories for train and validation sets if they don't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

# Function to move files from source to destination
def move_files(source_dir, dest_dir, file_list):
    for file in file_list:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

# Loop through each genre folder
for genre_folder in os.listdir(dataset_dir):
    genre_path = os.path.join(dataset_dir, genre_folder)
    if os.path.isdir(genre_path):
        # Create train and validation directories for each genre
        train_genre_dir = os.path.join(train_dir, genre_folder)
        validation_genre_dir = os.path.join(validation_dir, genre_folder)
        if not os.path.exists(train_genre_dir):
            os.makedirs(train_genre_dir)
        if not os.path.exists(validation_genre_dir):
            os.makedirs(validation_genre_dir)
        
        # List files in the genre folder
        files = os.listdir(genre_path)
        
        # Shuffle the files randomly
        random.shuffle(files)
        
        # Calculate the split index for train-validation split (80% train, 20% validation)
        split_index = int(0.8 * len(files))
        
        # Split files into train and validation sets
        train_files = files[:split_index]
        validation_files = files[split_index:]
        
        # Move train files
        move_files(genre_path, train_genre_dir, train_files)
        
        # Move validation files
        move_files(genre_path, validation_genre_dir, validation_files)

print("Train and validation sets created successfully.")
