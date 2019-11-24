# This is the main file for our CS1470 Final Project, group Going Merry! 
# Premise: Implementation of a CNN model to perform logo detection
# goal accuracy: >= 70%

# Note for team: KEEP SETTINGS AS "SPACES" & "4" FOR FORMATTING PLS

# CODE WILL BE REFACTORED LATER

import 
import tensorflow as tf
import numpy as np
from skimage.io import imread_collection
from keras.layers import Dense, Conv2D, Flatten

def preprocessing(train_file_path, test_file_path):
    
    train_dir = train_file_path
    train_labels = os.listdir(train_dir)
    train_imgs = imread_collection(train_dir)
    train_imgs = [tf.image.convert_image_dtype(i, dtype = tf.float32) for i in train_imgs] #convert each img to numbers
    train_imgs = [tf.convert_to_tensor(i) for i in train_imgs] #convert each img to a tensor
    
    test_dir = test_file_path
    test_labels = os.listdir(train_dir)
    test_imgs = imread_collection(test_dir)
    test_imgs = [tf.image.convert_image_dtype(i, dtype = tf.float32) for i in test_imgs] #convert each img to numbers
    test_imgs = [tf.convert_to_tensor(i) for i in test_imgs] #convert each img to a tensor
    
    return train_imgs, train_labels, test_imgs, test_labels

]

class Model(tf.keras.Model):
	def __init__(self):
		super(Model, self).__init__()

		self.learning_rate = 0.001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		# need to figure out best layer dimensions !_!
		# check all the layers, maybe add/remove some

		self.conv1 = tf.keras.layers.Conv2D(filters = , kernel_size = 3, strides = (2,2), padding = 'SAME')
		self.conv2 = tf.keras.layers.Conv2D(filters = , kernel_size = 3, strides = (2,2), padding = 'SAME')
		self.conv3 = tf.keras.layers.Conv2D(filters = , kernel_size = 3, strides = (2,2), padding = 'SAME')

		self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2)
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2)
		self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2)

		self.flatten = tf.keras.layers.Flatten()

		self.dense1 = tf.keras.layers.Dense()
		self.dense2 = tf.keras.layers.Dense()
		self.dense3 = tf.keras.layers.Dense()

    def call(self, inputs):
        conv_layer1 = self.pool1(self.conv1(inputs))
        conv_layer2 = self.pool2(self.conv2(layer1))
        conv_layer3 = self.pool3(self.conv3(layer2))
	
        flattened = self.flatten(conv_layer3)
	
        dense1 = self.dense1(flattened)
        dense2 = self.dense2(dense1)
        dense3 = self.dense3(dense2)

        return dense3
    
    def loss(self, prbs, labels):
        # will do
    def accuracy(self, prbs, labels):
	    # will do
        # aim for accuracy = 0.7
        
        print(accuracy)
        return accuracy

def train(model, train_inputs, train_labels):
    
    with tf.GradientTape() as tape:
	predictions = model.call(train_inputs)

def test(model, test_inputs, test_labels):
    prbs = model.call(test_inputs)
    return model.accuracy(prbs, test_labels)

# need to create method that will visually output the classification and accuracy
# this method will change depending on what we actually use as our testing inputs,
#     so it will end up being coded very very last 

def show_results(???):
#     Ideal goal: return image with visual indication of the location of logo being recognized
#     (or multiple, if there are multiple), maybe with a box ; ALSO want to have the accuracy of the
#     classification floating nearby

