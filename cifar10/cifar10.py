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

#用于描述CIFAR-10数据的全局变量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


