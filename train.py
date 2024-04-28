import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Print options
np.set_printoptions(threshold=sys.maxsize)

# Initialize constants
EPOCHS = 100
LR = 4e-3
BS = 64
DIMS = (299, 299, 3)

# Initialize the data and labels
data = []
labels = []
classes = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Romance", "Thriller"]

# Grab the image paths and randomly shuffle them
print("[INFO] loading images...")

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


print("[INFO] compiling model...")
# Load pre-trained Xception model without top classification layer
base_model = Xception(weights='imagenet', include_top=False, input_shape=DIMS)


# Freeze the base model
base_model.trainable = False

# Add custom classification layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(9, activation='sigmoid')  # 9 output classes for 9 genres
])

opt = Adam(learning_rate=LR)

# Compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# train the network
print("[INFO] training network...")
H = model.fit(
    x=train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("test.keras")

# plot the training loss and accuracy separately
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.title("Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plot.png")

# plot the training loss and accuracy separately
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_loss")
plt.title("Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("plot.png")

# plot the training loss and accuracy separately
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Validation Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("plot_val.png")

# print the classification report
print("[INFO] evaluating network...")
predictions = model.predict(valid_generator)
print(predictions)
print(valid_generator.classes)

