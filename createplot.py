# From the data copied from the terminal saved in a txt file, we can see that the model was saved to disk, and the mean absolute error, validation mean absolute error, training accuracy, validation accuracy, training f1-micro score, and validation f1-micro score were plotted.
# An example line from the ayo.txt file is:
# Epoch 1/80 - accuracy: 0.2455 - f1_macro: 0.2775 - f1_micro: 0.2890 - loss: 0.5129 - mae: 0.2898 - precision: 0.4352 - recall: 0.1924 - val_accuracy: 0.3401 - val_f1_macro: 0.3580 - val_f1_micro: 0.3944 - val_loss: 0.3960 - val_mae: 0.2559 - val_precision: 0.7339 - val_recall: 0.1646

import matplotlib.pyplot as plt
import numpy as np

# load the data from the txt file
with open("ayo.txt", "r") as file:
    data = file.readlines()

# extract the data
epochs = []
accuracy = []
f1_macro = []
f1_micro = []
loss = []
mae = []
precision = []
recall = []
val_accuracy = []
val_f1_micro = []
val_f1_macro = []
val_loss = []
val_mae = []
val_precision = []
val_recall = []

for line in data:
    line = line.split(" - ")
    epoch = line[0].split(" ")[1].split("/")[0]
    epochs.append(epoch)
    accuracy.append(float(line[1].split(": ")[1]))
    f1_macro.append(float(line[2].split(": ")[1]))
    f1_micro.append(float(line[3].split(": ")[1]))
    loss.append(float(line[4].split(": ")[1]))
    mae.append(float(line[5].split(": ")[1]))
    precision.append(float(line[6].split(": ")[1]))
    recall.append(float(line[7].split(": ")[1]))
    val_accuracy.append(float(line[8].split(": ")[1]))
    val_f1_macro.append(float(line[9].split(": ")[1]))
    val_f1_micro.append(float(line[10].split(": ")[1]))
    val_loss.append(float(line[11].split(": ")[1]))
    val_mae.append(float(line[12].split(": ")[1]))
    val_precision.append(float(line[13].split(": ")[1]))
    val_recall.append(float(line[14].split(": ")[1]))

# plot the accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), accuracy, label="accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./plots/plot-accuracy.png")

# plot the f1-macro score
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), f1_macro, label="f1_macro")
plt.title("F1-Macro Score")
plt.xlabel("Epoch #")
plt.ylabel("F1-Macro Score")
plt.legend()
plt.savefig("./plots/plot-f1-macro.png")

# plot the f1-micro score
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), f1_micro, label="f1_micro")
plt.title("F1-Micro Score")
plt.xlabel("Epoch #")
plt.ylabel("F1-Micro Score")
plt.legend()
plt.savefig("./plots/plot-f1-micro.png")

# plot the loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), loss, label="loss")
plt.title("Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./plots/plot-loss.png")

# plot the mean absolute error
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), mae, label="mae")
plt.title("Mean Absolute Error")
plt.xlabel("Epoch #")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.savefig("./plots/plot-mae.png")

# plot the precision
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), precision, label="precision")
plt.title("Precision")
plt.xlabel("Epoch #")
plt.ylabel("Precision")
plt.legend()
plt.savefig("./plots/plot-precision.png")

# plot the recall
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), recall, label="recall")
plt.title("Recall")
plt.xlabel("Epoch #")
plt.ylabel("Recall")
plt.legend()
plt.savefig("./plots/plot-recall.png")

# plot the validation accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), val_accuracy, label="val_accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.savefig("./plots/plot-val-accuracy.png")

# plot the validation f1-macro score
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), val_f1_macro, label="val_f1_macro")
plt.title("Validation F1-Macro Score")
plt.xlabel("Epoch #")
plt.ylabel("Validation F1-Macro Score")
plt.legend()
plt.savefig("./plots/plot-val-f1-macro.png")


# plot the validation f1-micro score
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), val_f1_micro, label="val_f1_micro")
plt.title("Validation F1-Micro Score")
plt.xlabel("Epoch #")
plt.ylabel("Validation F1-Micro Score")
plt.legend()
plt.savefig("./plots/plot-val-f1-micro.png")

# plot the validation loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), val_loss, label="val_loss")
plt.title("Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig("./plots/plot-val-loss.png")

# plot the validation mean absolute error
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), val_mae, label="val_mae")
plt.title("Validation Mean Absolute Error")
plt.xlabel("Epoch #")
plt.ylabel("Validation Mean Absolute Error")
plt.legend()
plt.savefig("./plots/plot-val-mae.png")

# plot the validation precision
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), val_precision, label="val_precision")
plt.title("Validation Precision")
plt.xlabel("Epoch #")
plt.ylabel("Validation Precision")
plt.legend()
plt.savefig("./plots/plot-val-precision.png")

# plot the validation recall
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), val_recall, label="val_recall")
plt.title("Validation Recall")
plt.xlabel("Epoch #")
plt.ylabel("Validation Recall")
plt.legend()
plt.savefig("./plots/plot-val-recall.png")

# plot the training and validation accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(epochs)), accuracy, label="train_accuracy")
plt.plot(np.arange(0, len(epochs)), val_accuracy, label="val_accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./plots/plot-train-val-accuracy.png")


