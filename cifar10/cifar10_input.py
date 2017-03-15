"""解码cifar10的二进制格式文件 
Routine for decoding the CIFAR-10 binary file format"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

#Process images of this size
IMAGE_SIZE = 24

#Global constants describing the CIFAR-10 data set
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
	"""从cifar10数据文件中读取和解析样本
	Reads and parses examples from CIFAR10 data files.
	Recommendation: if you want N-way read parallelism, call this function
	N times.  This will give you N independent Readers reading different
	files & positions within those files, which will give better mixing of
	examples.
	Args:
	filename_queue: A queue of strings with the filenames to read from.
	Returns:
	An object representing a single example, with the following fields:
	  height: number of rows in the result (32)
	  width: number of columns in the result (32)
	  depth: number of color channels in the result (3)
	  key: a scalar string Tensor describing the filename & record number
	    for this example.
	  label: an int32 Tensor with the label in the range 0..9.
	  uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""
	#Dimensions of the images in the CIFAR-10 dataset
	class CIFAR10Record(object):
		pass
	result = CIFAR10Record()

	label_bytes = 1
	result.height = 32
	result.width = 32
	result.depth = 3
	image_bytes = result.height * result.width * result.depth
	# Every record consists of a label followed by the image, with a
	# fixed number of bytes for each.
	record_bytes = label_bytes + image_bytes

	#读取一条数据，从filename_queue中获取文件名
	#cifar-10格式中没有头部和尾部，所以保持header-bytes和footer_bytes为0
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue)

	#将string转化为unint8类型的向量，与record_bytes等长
	record_bytes = tf.decode_raw(value, tf.uint8)

	#第一个字节代表label，转换类型uint8->int32
	result.label = tf.cast(
		tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

	#label之后的字节都代表图像
	#从原本的shape[depth*height*width]reshape为[height, width, depth]
	depth_major = tf.reshape(
		tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_bytes]),
		[result.depth, result.height, result.width])
	#从[depth, height, width]转置为[height, width, depth]
	#uint8image是一个[height, width, depth]shape的uint8格式的tensor
	result.uint8image = tf.transpose(depth_major, [1, 2, 0])

	return result


def _generate_image_and_label_batch(image, label, min_queue_examples, 
									batch_size, shuffle):
	"""构造一个图像和label的batch队列
	Construct a queued batch of images and labels.
	参数Args：
		image：3维tensor[height, width, 3]，数据类型type.float32
		label：1维tensor，数据类型type.int32
		min_queue_examples:int32, 队列中用于重训练的最少的样本数
		batch_size:每一个batch中的图像数
		shuffle:布尔值，代表是否使用乱序队列

	返回Returns：
		images:Images. 4维tensor，shape[batch_size, height, width, 3]
		labels:Labels. 1维tensor，shape[batch_size]
	"""
	#创建一个打乱了样本的队列，然后从队列中读取batch_size个(图像，label)对
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size)

	#在visualizer中显示训练图像
	tf.summary.image('images', images)

	return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
	"""使用Reader op为CIFAR训练构造augment data
	参数Args：
		data_dir：CIFAR-10数据目录路径
		batch_size：每个batch中的图像样本数

	返回Returns：
		images：Images. 4维tensor，shape[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
		labels：Labels. 1维tensor，shape[batch_size]
	"""
	filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	#创建一个队列用于读取文件名filenames
	filename_queue = tf.train.string_input_producer(filenames)

	#从filename_queue队列中的文件中读取样本
	read_input = read_cifar10(filename_queue)
	reshape_image = tf.cast(read_input.uint8image, tf.float32)

	
