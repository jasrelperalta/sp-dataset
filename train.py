import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

# Print options
np.set_printoptions(threshold=sys.maxsize)

# Initialize constants
EPOCHS = 1
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
train_csv_file_path = 'movie_dataset_train.csv'
valid_csv_file_path = 'movie_dataset_val.csv'
train_df = pd.read_csv(train_csv_file_path)
valid_df = pd.read_csv(valid_csv_file_path)

# Load the images and labels using image data generator from directory
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)

train_generator=datagen.flow_from_dataframe(
dataframe=train_df,
directory="newpo/posters",
x_col="filename",
y_col=classes,
batch_size=BS,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(299,299))

valid_generator=test_datagen.flow_from_dataframe(
dataframe=valid_df,
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
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.F1Score(average='macro', name='f1_macro'), tf.keras.metrics.F1Score(average='micro', name='f1_micro'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# train the network
print("[INFO] training network...")
H = model.fit(
    x=train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("./models/test.keras")

# plot the training f1-micro score
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["f1_micro"], label="train_f1_micro")
plt.plot(np.arange(0, EPOCHS), H.history["val_f1_micro"], label="val_f1_micro")
plt.title("Training F1-Micro Score")
plt.xlabel("Epoch #")
plt.ylabel("F1-Micro Score")
plt.legend()
plt.savefig("./plots/plot-f1-micro.png")

# plot the training f1-macro score
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["f1_macro"], label="train_f1_macro")
plt.plot(np.arange(0, EPOCHS), H.history["val_f1_macro"], label="val_f1_macro")
plt.title("Training F1-Macro Score")
plt.xlabel("Epoch #")
plt.ylabel("F1-Macro Score")
plt.legend()
plt.savefig("./plots/plot-f1-macro.png")


# plot the training precision
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["precision"], label="train_precision")
plt.plot(np.arange(0, EPOCHS), H.history["val_precision"], label="val_precision")
plt.title("Training Precision")
plt.xlabel("Epoch #")
plt.ylabel("Precision")
plt.legend()
plt.savefig("./plots/plot-precision.png")

# plot the training recall
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["recall"], label="train_recall")
plt.plot(np.arange(0, EPOCHS), H.history["val_recall"], label="val_recall")
plt.title("Training Recall")
plt.xlabel("Epoch #")
plt.ylabel("Recall")
plt.legend()
plt.savefig("./plots/plot-recall.png")

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./plots/plot-loss.png")


# print the classification report
print("[INFO] evaluating network...")
predictions = model.predict(valid_generator)
print(predictions)
print(valid_generator.classes)

