# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:34:16 2020

@author: Daniel

Figuring out Keras by implementing the same neural network that I hard-coded
for classifying the MNIST digit data-set. 3-layer net with tanh, then softmax
activation functions, followed by categorical cross-entropy loss.
"""
    
import tensorflow as tf

# set up train and test sets here
# using the 28x28 rather than the 8x8 MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# setting up the three-layer model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(20, activation='tanh'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# setting the loss function to categorical cross-entropy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# fit the network to the data
model.fit(x_train, y_train, epochs=10)

# evaluate the performance of the model
# seems like the accuracy for my from-scratch net was approximately the same
# about 97-98% :)
# this is much easier than coding it from scratch
model.evaluate(x_test,  y_test, verbose=2)