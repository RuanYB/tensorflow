"""构建CIFAR-10网络
函数简介：
	#计算用于训练的输入图像和label，
	#如果要对模型作评价(evaluation)，使用函数inputs()
	inputs, labels = distorted_inputs()

	#根据模型输入计算推断(inference)输出预测
	predictions = inference(inputs)

	#参照labels，计算预测的总损失量
	loss = loss(predictions, labels)

	#创建图(graph)，根据loss进行一次训练
	train_op = train(loss, global_step)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

#基本模块参数
tf.app.flags.DEFINE_integer('batch_size', 128,
							"""Number of images to process in a batch""")
tf.app.flags.DEFINE_string('data_dir', 'd:/Workspace/tensorflow/cifar10/cifar10_data',
							"""Path to the CIFAR-10 data directory""")
tf.app.flags.DEFINE_boolean('use_fp16', False, 
							"""Train the model using fp16""")

#用于描述CIFAR-10数据的全局常量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

#用于描述训练过程的常量
MOVING_AVERAGE_DECAY = 0.9999 #用于滑动平均的decay
NUM_EPOCHS_PER_DECAY = 350.0 #指定学习率在多少个epoch之后开始下降
LEARNING_RATE_DECAY_FACTOR = 0.1 #学习率的下降因子
INITIAL_LEARNING_RATE = 0.1 #初始学习率

#如果模型使用多GPU训练的话，需要在所有op名前加上tower_name来加以区分
#注意：这些前缀在可视化模型时会从summaries的名字中去掉
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
	"""帮助创建激活值的summaries的辅助函数(helper)

	创建一个summary用于绘制激活值的柱状图(histogram)
	创建一个summary用于评估激活值得稀疏性(sparsity)

	参数Args：
		x: Tensor
	返回Returns：
		无
	"""
	#从op名中去掉前缀'tower_[0-9]/'，防止这是一个多GPU训练任务
	#这样有助于在tensorboard中的表达清晰
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


#两种不同的变量创建方式，区别在于是否应用了权值衰减
def _variable_on_cpu(name, shape, initializer):
	"""帮助创建一个存储在cpu存储器中的变量的辅助函数(helper)

	参数Args:
		name: 变量名
		shape: 一个存储int类型数据的list
		initializer: 变量的初始化器
	返回Returns：
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""帮助创建一个应用权值衰减的初始化变量的辅助函数

	注意：变量的初始化服从truncated normal distribution
	权值衰减需要显式指定

	Args:
		name: 变量名
		shape: 存储int类型数据的list
		stddev: truncated Gaussian的标准差
		wd: 添加L2-Loss权值衰减并指定参数
	Returns：
		Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(name, shape, 
					tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def distorted_inputs():
	"""使用Reader op为CIFAR训练构造distorted input

	Returns:
		image: Images. 4维tensor，shape[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
		labels: Labels. 1维tenso，shape[batch_size]

	Raises:
		ValueError: 没有data_dir参数
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	images, labels = cifar10_input.distorted_inputs(data_dir, batch_size=FLAGS.batch_size)

	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels


def inputs(eval_data):
	"""使用Reader op为CIFAR评价构造输入数据

	Args:
		eval_data: 布尔值，表示是否使用训练或是评价数据集
	Returns:
		images: Images. 4维tensor，shape[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
		labels: Labels. 1维tenso，shape[batch_size]

	Raises:
		ValueError: 没有data_dir参数
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	images, labels = cifar10_input.inputs(eval_data=eval_data, 
											data_dir=data_dir, 
											batch_size=FLAGS.batch_size)

	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels


def inference(images):
	"""创建CIFAR-10模型

	Args：
		函数distorted_inputs()或inputs()返回的图像样本
	Returns：
		Logits
	"""
	#conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights', 
											shape=[5, 5, 3, 64],
											stddev=5e-2,
											wd=0.0)
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv1)

	#pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
							padding='SAME', name='pool1')

	#norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

	#conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											shape=[5, 5, 64, 64],
											stddev=5e-2,
											wd=0.0)
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv2)

	#norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

	#pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
							padding='SAME', name='pool2')

	#local3
	with tf.variable_scope('local3') as scope:
		#两个全连接层：展开所有单元，以便一次矩阵乘法解决
		reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', 
												shape=[dim, 384],
												stddev=0.04, 
												wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local3)

	#local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights',
												shape=[384, 192],
												stddev=0.04,
												wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local4)

	#线性层(WX + b)
	#在这里并没有使用softmax，因为tf.nn.sparse_softmax_cross_entropy_with_logits
	#接受unscaled logits，函数内部实现了softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', 
												shape=[192, NUM_CLASSES],
												stddev=1/192.0,
												wd=0.0)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear


def loss(logits, labels):
	"""添加L2 Loss到所有的可训练变量上
	添加"Loss"和"Loss/avg"到summary中

	Args:
		logits: 函数inference()得来的logits
		labels: 函数distorted_inputs()或inputs()得来的labels，1维tensor，shape[batch_size]
	Returns:
		float类型的Loss tensor
	"""
	#计算一个batch中的样本的平均交叉熵损失
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection("losses", cross_entropy_mean)

	#总损失 = 交叉熵损失 + 所有的权重衰减项(L2 loss)
	#实际上只有local3和local4产生了有效衰减项，因为conv1、conv2和softmax的factor为0.0
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
	"""添加CIFAR-10模型的losses到summaries中

	为所有的损失生成滑动平均数，为网络性能可视化生成相关的summary

	Args:
		total_loss: 函数loss()获得的总损失
	Returns:
		loss_averages_op: 用于生成损失的滑动平均数的op
	"""
	#为所有独立的losses和total loss计算moving average
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	#为所有独立的losses、total loss和losses的平均数附上标量summary
	for l in losses+[total_loss]:
		#将每一个loss命名为(raw)，loss平均数则依然为原来的名字
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op


def train(total_loss, global_step):
	"""训练CIFAR-10模型

	创建一个optimizer并应用到所有的可训练参数上，为所有可训练参数加上滑动平均值

	Args:
		total_loss: 函数loss()获得的总损失
		global_step: 用于记录当前训练步数的Integer变量
	Returns:
		train_op: 用于训练的op
	"""
	#影响学习率的变量
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	#根据步数指数衰减学习率
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	tf.summary.scalar('learning rate', lr)

	#生成所有loss的滑动平均数奖金和相关的summary
	loss_averages_op = _add_loss_summaries(total_loss)

	#计算梯度
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		# Variable is always present, but gradient can be None.
		grads = opt.compute_gradients(total_loss)

	#应用梯度
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	#为可训练参数添加柱状图
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	#为梯度添加柱状图
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad) #记得这里留意下梯度在board中的名字是什么

	#监测所有可训练参数的滑动平均数
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([variables_averages_op]):
		train_op = tf.no_op(name='train')  #就返回一个operation？u kidding me？

	return train_op

#这段直接复制黏贴的
def maybe_download_and_extract():
	"""Download and extract the tarball from Alex's website."""
	dest_directory = FLAGS.data_dir
  	if not os.path.exists(dest_directory):
    	os.makedirs(dest_directory)
  	filename = DATA_URL.split('/')[-1]
  	filepath = os.path.join(dest_directory, filename)
  	if not os.path.exists(filepath):
    	def _progress(count, block_size, total_size):
      		sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          		float(count * block_size) / float(total_size) * 100.0))
      		sys.stdout.flush()
    	filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    	print()
    	statinfo = os.stat(filepath)
    	print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  	extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  	if not os.path.exists(extracted_dir_path):
		tarfile.open(filepath, 'r:gz').extractall(dest_directory)