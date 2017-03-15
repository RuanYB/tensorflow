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
	建议: 如果需要N条路并行读取数据，调用该函数N次，将获得N个独立的Reader用于
	读取这些文件中不同的文件和位置，样本的混合效果也会更好。
	参数Args:
		filename_queue: 一个string类型的队列，存储需要读取的文件名
	返回Returns:
		返回一个对象，代表一个独立的样本，包含以下field：
		height: result中的行数(32)
		width: result中的列数(32)
		depth: result中的颜色通道数(3)
		key: 一个string类型的标量Tensor，描述样本的文件名和记录编号
		label: 一个int32类型的Tensor，代表样本label，取值范围0~9
		uint8image: 一个uint8类型，shape[height, width, depth]的Tensor，描述图像数据
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
	"""使用Reader op为CIFAR训练环节构造augment data
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
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	#处理用于训练的图像，对图像作不同的变形
	
	#图像随机裁剪成[height, width]的大小
	distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

	#随机水平翻转图像
	distorted_image = tf.image.random_flip_left_right(distorted_image)

	#随机改变亮度和对比度
	distorted_image = tf.image.random_brightness(distorted_image, 
												max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image, 
												lower=0.2, upper=1.8)

	#标准化：每个像素减去平均值，除以方差
	float_image = tf.image.per_image_standardization(distorted_image)

	#调整tensor的shape
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	#确保随机打乱有着良好的混合性质
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
							min_fraction_of_examples_in_queue)
	print('Filling queue with %d CIFAR images before starting to train'
			'This will take a few minutes.' % min_queue_examples)

	#创建样本队列，生成一个batch的images & labels
	return _generate_image_and_label_batch(float_image, read_input.label, 
											min_queue_examples, batch_size, 
											shuffle=True)


def inputs(eval_data, data_dir, batch_size):
	"""使用Reader op为CIFAR评价环节构造输入
	参数Args：
		eval_data:布尔值，代表是否使用训练或者评价数据集
		data_dir:CIFAR-10数据目录路径
		batch_size:每一个batch的图像数
	返回Returns：
		images:Images. 4维tensor，shape[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
		labels:Labels. 1维tensor
	"""
	if not eval_data:
		filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
					for i in xrange(1,6)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	#创建一个队列用于读取文件名
	filename_queue = tf.train.string_input_producer(filenames)

	#从filename队列的文件中读取样本
	read_input = read_cifar10(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	#处理用于评价的图像
	#图像中心裁剪为[height, width]的大小
	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 
															width, height)

	#每个像素减去平均值，除以方差
	float_image = tf.image.per_image_standardization(resized_image)

	#调整tensor的shape
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	#确保随机打乱有着良好的性质
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
							min_fraction_of_examples_in_queue)

	#创建样本队列，生成一个batch的images & labels
	return _generate_image_and_label_batch(float_image, read_input.label,
											min_queue_examples, batch_size,
											shuffle=False)