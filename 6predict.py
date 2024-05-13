
from keras.saving import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
import numpy as np
import sys



# Print options
np.set_printoptions(threshold=sys.maxsize)

# Initialize constants
EPOCHS = 100
LR = 4e-3
BS = 64
DIMS = (299, 299, 3)
classes = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Romance", "Thriller"]

# Load CSV file
csv_file_path = 'movie_dataset_combined.csv'
dataframe = pd.read_csv(csv_file_path)

# Load the images and labels using image data generator from directory
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)

train_generator=datagen.flow_from_dataframe(
dataframe=dataframe[:(len(dataframe) // 5) * 4],
directory="newpo/posters",
x_col="filename",
y_col=classes,
batch_size=BS,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(299,299))

valid_generator=test_datagen.flow_from_dataframe(
dataframe=dataframe[(len(dataframe) // 5) * 4:],
directory="newpo/posters",
x_col="filename",
y_col=classes,
batch_size=BS,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(299,299))

# print("[INFO] getting true values...")
# y_true = []
# for x,y in valid_generator:
#     y_true.append(y)

#Load model
model = load_model(filepath='test.keras', compile=False, custom_objects={'input_shape': DIMS})

# Get the F1 score
print("[INFO] getting F1 score...")
y_pred = model.predict(valid_generator)
y_true = []
for x,y in valid_generator:
    y_true.append(y)
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred = np.round(y_pred)
f1 = tf.keras.metrics.F1Score(y_true, y_pred, average='macro')
print(f1)


