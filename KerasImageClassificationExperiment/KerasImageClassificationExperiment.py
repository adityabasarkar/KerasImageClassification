# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
plt.figure(figsize=(10,5))
for i in range(12):
    plt.subplot(2,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""

train_images = train_images / 255.0
#print(train_images[0])

test_images = test_images / 255.0

model = tf.keras.Sequential()

"""
Build Model
----------------------------------------------
"""

# Feature Extraction Layer 1
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 32 features maps: Each 26 x 26
print("After First Convolution:", model.output_shape)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# 32 max pooled maps: Each 13 x  13
print("After First Max Pooling:", model.output_shape)

# Feature Extraction Layer 2
model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
# 32 feature maps: Each 12 x 12
print("After Second Convolution:", model.output_shape)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# 32 max pooled maps: Each 6 x 6
print("After Second Max Pooling:", model.output_shape)

# DNN 
model.add(tf.keras.layers.Flatten())
print("Flattened input:", model.output_shape)
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])



model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc, '\nTest Loss:', test_loss)