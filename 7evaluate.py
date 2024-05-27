from keras.saving import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

# Print options
np.set_printoptions(threshold=sys.maxsize)

# Initialize constants
LR = 1e-4
BS = 64
DIMSXC = (299, 299, 3)
DIMSVGG = (224, 224, 3)

# Initialize the data and labels
data = []
labels = []
classes = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Romance", "Thriller"]

# Get arguments
modelName = sys.argv[1]

# Load Model
modelPath = f'models/{modelName}.keras'
model = load_model(filepath=modelPath)

# Load CSV file
test_csv_file_path = 'movie_dataset_test2.csv'
test_dataframe = pd.read_csv(test_csv_file_path)

# Load the images and labels using image data generator from directory
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input,)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_dataframe,
    directory="combined_posters",
    x_col="filename",
    y_col=classes,
    batch_size=BS,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(299,299))

# Evaluate the model
print("[INFO] evaluating model...")
results = model.evaluate(test_generator)
print(results)

# Save the results
with open(f"teststats/{modelName}-test.txt", "w") as file:
    file.write(f"Loss: {results[0]}\n")
    file.write(f"Accuracy: {results[1]}\n")
    file.write(f"F1-Macro: {results[2]}\n")
    file.write(f"F1-Micro: {results[3]}\n")
    file.write(f"Precision: {results[4]}\n")
    file.write(f"Recall: {results[5]}\n")
    file.write(f"MAE: {results[6]}\n")
    file.write(f"Val Loss: {results[7]}\n")
    file.write(f"Val Accuracy: {results[8]}\n")
    file.write(f"Val F1-Macro: {results[9]}\n")
    file.write(f"Val F1-Micro: {results[10]}\n")
    file.write(f"Val Precision: {results[11]}\n")
    file.write(f"Val Recall: {results[12]}\n")
    file.write(f"Val MAE: {results[13]}\n")

