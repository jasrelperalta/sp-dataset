import os
import shutil

# File path
photos = 'combined_posters/'
csv_file_path = 'movie_dataset_combined2.csv'

# Check if directory exists
if not os.path.exists("testsplit"):
    os.makedirs("testsplit")

# Copy every 10th image to testsplit folder
with open(csv_file_path, 'r') as file:
    # skip the header
    next(file)
    for i, line in enumerate(file):
        if i % 10 == 0:
            filename = line.split(',')[0]
            print(f"Copying {filename} to testsplit")
            shutil.copy(photos + filename, 'testsplit/' + filename)

print("Test split created successfully")