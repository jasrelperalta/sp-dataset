import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import os


# Load the all models in the models directory
for model in os.listdir("models/finalmodels"):
    modelPath = f'models/finalmodels/{model}'
    modelName = model.split(".")[0]

    # Load the model
    model = load_model(filepath=modelPath)

    # print the model summary
    print(modelName)
    print(model.summary())
    print("\n\n")
