"""
a convolutional MNIST classifier
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
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	#functions used for initialize weight and biases
	#initialize weights with a small amount of noise
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	#initialize ReLU neurons with a slightly positive initial bias to avoid "dead neurons"
	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	#design the convolutional layer
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#first convolutional layer
	#weights: [patch_size, num_input_channel, num_output_channel]
	#bias: [num_output_channel]
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	#tf.reshape(tensor, shape, name=None): 将tensor转变为shape的形式
	#只允许shape中一项为-1，表示无需指定该维大小，自动计算
	x_image = tf.reshape(x, [-1, 28, 28, 1])


	'''
	convolve x_image with the weight tensor, 
	add the bias, 
	apply the ReLU function, 
	and finally max pool
	'''
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	#reduce the image size to 14x14
	h_pool1 = max_pool_2x2(h_conv1)

	#second convolutional layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	#reduce the image size to 7x7
	#打印h_pool2的size
	h_pool2 = max_pool_2x2(h_conv2)


	#densely connected layer
	'''
	 reshape the tensor from the pooling layer into a batch of vectors, 
	 multiply by a weight matrix, 
	 add a bias, 
	 and apply a ReLU
	'''
	#add a fully-connected layer with 1024 neurons to allow processing on the entire image
	W_fc1 = weight_variable([7*7*64, 1024])
	b_fc1 = bias_variable([1024])

	#打印h_pool2_flat的size
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


	'''
	dropout
	create a placeholder for the probability that a neuron's output is kept during dropout,
	tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking
	'''
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	#readout layer
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	'''
	train and evaluate the model
	'''
	#loss function
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for i in range(20000):
		batch = mnist.train.next_batch(50)
		#每100个epoch打印一次
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x:batch[0], y_:batch[1], keep_prob:1.0})
			print("step %d, training accuracy %g" % (i, train_accuracy))
		train_step.run(feed_dict={
			x:batch[0], y_:batch[1], keep_prob:0.5})

	print("test accuracy: %g" % accuracy.eval(feed_dict={
		x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='d:/Workspace/tensorflow/MNIST_data',
						help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	#FLAG中为parser的解析的关键字参数，unparsed中为非关键字参数
	#sys.argv中依照键入顺序存储python命令之后的所有参数
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)