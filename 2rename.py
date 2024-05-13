import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if '_' in filename:
            new_filename = filename.split('_', 1)[1]
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

# Replace 'folder_path' with the path to this folder
# Iterate through all files in the folder and rename them
folder_path = 'posters2/'
for folder in os.listdir(folder_path):
    rename_files(os.path.join(folder_path, folder))
