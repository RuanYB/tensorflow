"""
a very simple MNIST classifier
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):
	#import data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	#create nodes for the input images and target output classes
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	#define the weights w and biases b for model
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	#implement regression model
	#multiply the vectorized input images x by the weight matrix W, add the bias b
	y = tf.matmul(x, W) + b
	
	#specify a loss function
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	#train the model
	#minimize cross entropy using gradient descent with a learning rate of 0.5
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()

	#initialize variable with specified values
	tf.global_variables_initializer().run()


	#run the training procedure
	for _ in range(1000):
		batch = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

	#evaluate the model 
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='d:/Workspace/tensorflow/MNIST_data',
						help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
