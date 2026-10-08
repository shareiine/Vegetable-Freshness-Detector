# PROCESS: pre-process, architecture, train, and test

# USAGE: python cnn.py --dataset dataset

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


# construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = 
	"parth to the input dataset")
args = vars(ap.parse_args())

# list of images that will be described
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# load the dataset (images)
data = []
labels = []
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))
	image = image / 255.0
	data.append(image)

	# for the labels, get the folder name 
	label = os.path.basename(os.path.dirname(imagePath))
	labels.append(label)

x = np.array(data)
y = np.array(labels)

# split the data
(trainX, testX, trainY, testY) = train_test_split(x, y)

# change images to [0 - 1] format
# trainX = trainX.astype("float32") / 255.0
# testX = testX.astype("float32") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Define learning rate schedules since decay has been deprecated
lr_schedule = ExponentialDecay(
	initial_learning_rate = 0.01,
	decay_steps = 100,
	decay_rate = 0.9)

# Early stopping function to avoid overfitting
early_stop = EarlyStopping(
	monitor = "val_loss",
	patience = 15,
	restore_best_weights = True)


# get the optimizer
print("[INFO] compiling model...")
opt = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

# initialize the model
classes = np.unique(labels).shape[0]
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=classes)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size = 64, epochs = 40, callbacks = [early_stop])

# get the number of epochs
epochs = len(H.history["loss"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)

# create folder to store figure
if not os.path.exists("output"):
	os.makedirs("output")
	print("[INFO] creating output folder...")
output_path = os.path.join("output", "plot.png")

# save model
model.save("vegetable_detector.h5")  
print("[INFO] model saved as vegetable_detector.h5")


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Vegetable Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(output_path)

