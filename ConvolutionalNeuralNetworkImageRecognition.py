import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# load cifar10 dataset, normalize data by dividing by 255, RGB values end up being b/w 0 and 1
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
# next define the possible class names in a list to label resuts later
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# let's visualize a section of the data
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
# run 16 iterations to create a 4x4 grid of subplots, leave x and y ticks empty
# if your computer is slow, don't use all of the data
train_images = train_images[:20000]
train_labels = train_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]
# now that we prepared our data, lets build the neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation ='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# inputs got into Conv2D layer with 32 filters in the shape of 3x3 matricies
# activation relu with input shape of 32x32x3, because images have resolution of 32x32 w/ 3 layers because RGB colors
# evntually flattened into a 1 dimensional vector
# lets train and test the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
# epochs of 10 means the model will see the data 10 times
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# now we can feed images to the algorithm and have them predict what the image contains
# we have to get images down to 32x32 pixels, you can use Gimp or Paint to crop and/or scale them
# we will load the images into our script, using OpenCV
img1 = cv.imread('car.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.imread('horse.jpg')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
plt.imshow(img1, cmap=plt.cm.binary)
plt.show()
# load your image file names, use cvtColor to change default BGR to RGB
plt.imshow(img1, cmap=plt.cm.binary)
plt.show()
# we can use the images for the input for our model to get a prediction
prediction = model.predict(np.array([img1]) / 255)
index = np.argmax(prediction)
print(class_names[index])
# have to normailze the image and use argMax to reveal the guess with the highest probability
