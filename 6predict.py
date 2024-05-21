
from keras.saving import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
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
modelPath = 'models/may14test.keras'
classes = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Romance", "Thriller"]

def initialize():
    # Load CSV file
    csv_file_path = 'movie_dataset_combined2.csv'
    test_csv_file_path = 'movie_dataset_test2.csv'
    dataframe = pd.read_csv(csv_file_path)
    test_dataframe = pd.read_csv(test_csv_file_path)

    # Load the images and labels using image data generator from directory
    datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)

    # Load the images and labels using image data generator from directory
    generator=datagen.flow_from_dataframe(
    dataframe=dataframe,
    directory="combined_posters",
    x_col="filename",
    y_col=classes,
    batch_size=BS,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(299,299))

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

    #Load model
    model = load_model(filepath=modelPath, compile=True, custom_objects={'input_shape': DIMS})

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model and the generator
    return model, generator, test_generator

# Classify a single image
def classify_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array, verbose=1)

    print(predictions)
    return predictions

# Classify a batch of images
def classify_batch(batch, model):    
    predictions = model.predict(batch)
    print(predictions)
    return predictions


if __name__ == "__main__":
    model, generator, test_generator = initialize()

    if len(sys.argv) > 1:
        classify_image(sys.argv[1], model)
