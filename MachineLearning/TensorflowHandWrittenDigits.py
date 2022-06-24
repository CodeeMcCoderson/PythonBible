import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv #computer vision
# import data provided by tensorflow. Consists of 70,000 images of handwritten digits w/ resolution of 28x28 pixels
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# returns 2 tuples
# data is stored in NumPy arrays, need to normalize
X_train = tf.keras.utils.normalize(X_train)
X_test- tf.keras.utils.normalize(X_test)
# we dont sacle results because they should be values 0-9
# contruct the neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
# Sequential is linear model defined layer after layer
# First layer we add is a flatten layer to make the 28x28 a 784x1
# Next add two hidden layers with 128 neaurons a piece, 'relu' is a rectified linear unit
# Last layer has 10 neaurons (0-9) and softmax makes all outputs add up to 1, giving us 10 different probabilities
# Lets compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# optimizer and loss function need to read documentation to understand
model.fit(X_train, Y_train, epochs=3)
loss, accuracy = model.evaluate(X_test, Y_test)
print("Loss: ", loss)
print("Accuracy ", accuracy)
# epochs is how many times we re-run the data through
# accuracy is percentage of numbers guessed correctly, loss is summation of errors made
#pip install opencv-python
#image = cv.imread('digit.png')(:,:,0)
#image = np.invert(np.array([image]))
#remove dimention, convert to NumPy array and invert
#prediction = model.predict(image)
#print("Prediction:{}".format(np.argmax(prediction)))
#plt.imshow(image[0])
#plt.show()
# prediction is a n array of 10 different probabilities, argmax shows highest probability
