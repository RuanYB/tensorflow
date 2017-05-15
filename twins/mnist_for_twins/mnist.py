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
import mnist_input
import os

MIN_AFTER_DEQUEUE = 1000
DIR_TFRECORDS = "data"

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
		var = tf.get_variable(name, shape, dtype=tf.float32, initializer=init)
	# init = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
	# var = tf.get_variable(name, shape, dtype=tf.float32, initializer=init)
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
		var = tf.get_variable(name, shape, initializer=init, dtype=tf.float32)
	# init = tf.constant_initializer(0.1, dtype=tf.float32)
	# var = tf.get_variable(name, shape, initializer=init, dtype=tf.float32)
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
	return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(input):
	"""
	池化层处理函数
	Args:
		input: 池化层输入tensor
	Returns:
		池化层tensor
	"""
	return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inputs(data_type):
	"""
	加载训练/测试数据集
	Args:
		datatype: 获取训练集batch还是测试集batch，'train/test'
	Returns:
		image_batch: 一个batch size(50)的图片样本
		label_batch: 一个batch size(50)的label样本
	"""
	shuffle = True
	file_path = os.path.join(DIR_TFRECORDS, '%s.tfrecords' % data_type)
	if not os.path.exists(file_path):
		mnist_input.generate_tfrecords(data_type)
	if 'test' == data_type:
		shuffle = False
	image, label = mnist_input.parse_tfrecords(tfrecord_type=data_type)
	return mnist_input.generate_image_and_label_batch(
						image, label, MIN_AFTER_DEQUEUE, batch_size=50, shuffle=shuffle)


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
		pre_activation = tf.nn.bias_add(conv2d(images, kernal), biases);
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	#第一个池化层
	pool1 = max_pool(conv1)

	#第二个卷积层
	with tf.variable_scope('conv2') as scope:
		kernal = weight_variable('weights', shape=[5, 5, 32, 64])
		biases = bias_variable('biases', shape=[64])
		pre_activation = tf.nn.bias_add(conv2d(pool1, kernal), biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	#第二个池化层
	#7*7*64
	pool2 = max_pool(conv2)

	#全连接层
	with tf.variable_scope('fc1') as scope:
		weight_fc1 = weight_variable('weights', shape=[7*7*64, 1024])
		biases = bias_variable('biases', shape=[1024])
		pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
		fc1 = tf.nn.relu((tf.matmul(pool2_flat, weight_fc1) + biases), name=scope.name)

	fc1_drop = tf.nn.dropout(fc1, keep_prob=dropout)
	print(dropout) #for test

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
		tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
	return cross_entropy

def train_accuracy(logits, labels):
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

def train(cross_entropy, global_step):
	# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	opt = tf.train.AdamOptimizer(1e-4)
	grads = opt.compute_gradients(cross_entropy)
	train_op = opt.apply_gradients(grads, global_step=global_step)
	return train_op