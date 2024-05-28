from keras.saving import load_model # type: ignore
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os

# Print options
np.set_printoptions(threshold=sys.maxsize)

# Class names
classes = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Romance", "Thriller"]

# Load sys arguments
imagePath = sys.argv[1]
modelName = sys.argv[2]

# if model has xception in its name, preprocess the image for xception
modelMode = "xception" if "xception" in modelName else "vgg16"

# Load Model
modelPath = f'models/finalmodels/{modelName}.keras'
model = load_model(filepath=modelPath)

# Load the image
if modelMode == "xception":
    image = tf.keras.preprocessing.image.load_img(imagePath, target_size=(299, 299))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.xception.preprocess_input(image)
else:
    image = tf.keras.preprocessing.image.load_img(imagePath, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)

# Predict the image
prediction = model.predict(image)

# Print the prediction with the class names
for i in range(len(classes)):
    print(f"{classes[i]}: {prediction[0][i]}")