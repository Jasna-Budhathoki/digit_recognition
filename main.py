import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing 

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

#Check for loss and accuracy on a test set 

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

#Prediction 

image_number = 1
while os.path.isfile(f"pics/pic{image_number}.png"):
    img = cv2.imread(f"pics/pic{image_number}.png")[:,:,0]
    img = np.invert(np.array([img]))
    print(img.shape)
    prediction = model.predict(img)
    print(f"The digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap = plt.cm.binary)
    plt.show()
    image_number += 1







