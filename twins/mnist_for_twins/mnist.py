# encoding： utf-8
"""
@author: ryan ruan
@contact: wufazhuceryan@yeah.net

@version: 1.0
@license: Apache Licence
@file: mnist_input.py
@time: 2017/4/23 22:49

构建mnist网络
"""
import tensorflow as tf
import numpy as np

from mnist_input import load_train_images, load_train_labels, load_test_images, load_test_labels, DataSet


def weight_variable(name, shape):
	"""
	权重变量的创建与初始化

	Args:
		name: 变量名
		shape: 变量shape
	Returns:
		var: 权重变量
	"""
	with tf.device('/gpu:0'):
		init = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
		var = tf.get_variable(name, shape, init, dtype=tf.float32)
	return var

def bias_variable(name, shape):
	"""
	偏置变量的创建与初始化

	Args:
		name: 变量名
		shape: 变量shape
	Returns:
		var: 偏置变量
	"""
	with tf.device('/gpu:0'):
		init = tf.constant_initializer(0.1, dtype=tf.float32)
		var = tf.get_variable(name, shape, init, dtype=tf.float32)
	return var


def conv2d(input, filter):
	"""
	创建卷积层的再封装函数
	Args:
		input: 该层输入tensor
		filter: 卷积核, [width, height, input_channel, output_channel]
	Returns:
		卷积层tensor
	"""
	return tf.nn.con2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(input):
	"""
	池化层处理函数
	Args:
		input: 池化层输入tensor
	Returns:
		池化层tensor
	"""
	return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inputs_train():
	"""
	加载训练数据集
	Args:
		None
	Returns:
		DataSet: 数据集对象，调用next_batch()方法返回(images, labels)
	"""
	images = load_train_images()
	labels = load_train_labels()
	data = DataSet(images, labels)
	return data 

def inputs_eval():
	"""
	加载评价数据集
	Args:
		None
	Returns:
		DataSet: 数据集对象，调用next_batch()方法返回(images, labels)
	"""
	images = load_test_images()
	labels = load_test_labels()
	data = DataSet(images, labels)
	return data


#创建mnist卷积模型
def inference(images, dropout=1.0):
	"""
	Args:
		images: inputs()函数返回的数据集对象的images部分
	Returns:
		Logits
	"""
	#第一个卷积层
	with tf.variable_scope('conv1') as scope:
		kernal = weight_variable('weights', shape=[5, 5, 1 ,32])
		biases = bias_variable('biases', shape=[32])
		pre_activation = tf.nn.bias_add(conv2d(imags, kernal), biases);
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	#第一个池化层
	pool1 = max_pool(conv1)

	#第二个卷积层
	with tf.variable_scope('conv2') as scope:
		kernal = weight_variable('weights', shape=[5, 5, 32, 64])
		biases  =bias_variable('biases', shape=[64])
		pre_activation = bias_add(conv2d(pool1, kernal), biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	#第二个池化层
	pool2 = max_pool(conv2)

	#全连接层
	with tf.variable_scope('fc1') as scope:
		weight_fc1 = weight_variable('weights', shape=[7*7*64, 1024])
		biases = bias_variable('biases', shape=[1024])
		pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
		fc1 = tf.nn.relu((tf.matmul(pool2_flat * weight_fc1) + biases), name=scope)

	fc1_drop = tf.nn.dropout(fc1, dropout=dropout)

	#输出层
	with tf.variable_scope('softmax_linear') as scope:
		weight_fc2 = weight_variable('weight', shape=[1024, 10])
		biases = bias_variable('biases', shape=[10])

		softmax_linear = tf.add(tf.matmul(fc1_drop, weight_fc2), biases, name=scope.name)

	return softmax_linear


def loss(logits, labels):
	"""
	Args:
		logits: 函数inference()得来的logits
		labels: 函数inputs()得来的labels
	Returns:
		float类型的loss tensor
	"""
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
	return cross_entropy


def train(cross_entropy):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	return train_step