# encoding: utf-8
"""
@author: ryan ruan
@contact: wufazhuceryan@yeah.net

@version: 1.0
@license: Apache Licence
@file: mnist_input.py
@time: 2017/4/20 14:59

转换MNIST手写数字数据文件转换为bmp图片文件格式

=====================
IDX文件格式的解析规则：
=====================
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf

# 训练集文件
TRAIN_IMAGES_PATH = 'data/train-images.idx3-ubyte'
# 训练集标签文件
TRAIN_LABELS_PATH = 'data/train-labels.idx1-ubyte'
# 测试集文件
TEST_IMAGES_PATH = 'data/t10k-images.idx3-ubyte'
# 测试集标签文件
TEST_LABELS_PATH = 'data/t10k-labels.idx1-ubyte'


#=============================================DECODER FUNCTIONS===============================================
def decode_idx3_ubyte(idx3_ubyte_file):
	"""
	解析idx3文件的通用函数
	:param idx3_ubyte_file: idx3文件路径
	:return: 数据集
	"""
	#读取二进制数据
	bin_data = open(idx3_ubyte_file, 'rb').read()

	#解析文件头信息，依次为魔数（magic number）、图片数量、图片高、图片宽
	offset = 0
	fmt_header = '>iiii'
	magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
	print('魔数:%d, 图片数量:%d张, 图片大小:%d*%d' % (magic_number, num_images, num_cols, num_rows))

	#解析数据集
	image_size = num_rows * num_cols
	offset += struct.calcsize(fmt_header)
	fmt_image = '>' + str(image_size) + 'B'
	images = np.empty((num_images, num_rows, num_cols))
	for i in range(num_images):
		if((i+1) % 10000 == 0):
			print('已解析 %d 张图片' % (i+1)) 
		images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
		offset += struct.calcsize(fmt_image)
	return images.astype(int)


def decode_idx1_ubyte(idx1_ubyte_file):
	"""
	解析idx1文件的通用函数
	:param idx1_ubyte_file: idx1文件路径
	:return: 数据集
	"""
	#解析出的label是numpy模块默认的float64类型的数据
	#读取二进制数据
	bin_data = open(idx1_ubyte_file, 'rb').read()

	#解析头文件信息，依次为魔数、标签数
	offset = 0
	fmt_header = '>ii'
	magic_number, num_labels = struct.unpack_from(fmt_header, bin_data, offset) 
	print('魔数:%d, 标签数量:%d' % (magic_number, num_labels))
	
	offset += struct.calcsize(fmt_header)
	fmt_label = '>B'
	lables = np.empty(num_labels)
	for i in range(num_labels):
		if((i+1) % 10000 == 0):
			print('已解析 %d 张' % (i+1))

		lables[i] = np.array(struct.unpack_from(fmt_label, bin_data, offset))
		offset += struct.calcsize(fmt_label)
	return lables.astype(int)


#==========================================TFRecords RELATED=================================================
#创建tf.train.Feature的helper函数
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords(dataset_type):
	"""
	生成包含tf.train.Example protocol buffers (包含特征Features域)的TFRecords文件：
		step1: 使用tf.train.Example（协议内存块protocol buffer）定义需要填入的数据格式，协议内存块包含字段Features
		step2: 然后使用tf.python_io.TFRecordWriter来写入
	"""
	if 'train' == dataset_type:
		images_dir = TRAIN_IMAGES_PATH
		labels_dir = TRAIN_LABELS_PATH
	elif 'test' == dataset_type:
		images_dir = TEST_IMAGES_PATH
		labels_dir = TEST_LABELS_PATH
	else:
		raise ValueError('please select correct dataset_type type: train or test')

	#解析mnist数据集文件
	images = decode_idx3_ubyte(images_dir)
	labels = decode_idx1_ubyte(labels_dir)

	rows = images.shape[1]
	cols = images.shape[2]
	depth = 1

	#将每个（image, label）对写入Example的Feature属性中，构造TFRecords集
	writer = tf.python_io.TFRecordWriter("data/%s.tfrecords" % dataset_type) 
	for index in range(images.shape[0]):
		if index % 1000 == 0:
			print('已经完成存储 %d 张图片的TFRecords处理...' % index)
		#将每一个图片扁平化之后，转换为字节数据
		#image_raw = images[index].ravel().tostring()
		image_raw = images[index].tobytes()
		#创建协议内存块
		example = tf.train.Example(features=tf.train.Features(feature={
			'height' : _int64_feature(rows),
			'width' : _int64_feature(cols),
			'depth' : _int64_feature(depth),
			'labels' : _int64_feature(labels[index]),
			'image_raw' : _bytes_feature(image_raw)
			}))
		#将example写入TFRecord文件中
		writer.write(example.SerializeToString())
	#关闭writer
	writer.close()


def parse_tfrecords(tfrecord_type):
	"""
	读取TFRecords的函数
	Args:
		TFRecords格式数据文件
	"""
	#根据文件名生成一个队列
	if 'train' == tfrecord_type:
		images_dir = TRAIN_IMAGES_PATH
		labels_dir = TRAIN_LABELS_PATH
	elif 'test' == tfrecord_type:
		images_dir = TEST_IMAGES_PATH
		labels_dir = TEST_LABELS_PATH
	else:
		raise ValueError('please select correct tfrecord_type type: train or test')
	filename = "data/%s.tfrecords" % tfrecord_type
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()

	#返回文件名和文件
	_, serialized_example = reader.read(filename_queue)
	#解析feature域信息
	features = tf.parse_single_example(serialized_example, features={
		'height' : tf.FixedLenFeature([], tf.int64),
		'width' : tf.FixedLenFeature([], tf.int64),
		'depth' : tf.FixedLenFeature([], tf.int64),
		'labels' : tf.FixedLenFeature([], tf.int64),
		'image_raw' : tf.FixedLenFeature([], tf.string)
		})
	rows = tf.cast(features['height'], tf.int32)
	cols = tf.cast(features['width'], tf.int32)

	image = tf.reshape(tf.decode_raw(features['image_raw'], tf.int32), [28, 28])
	image = tf.cast(image, tf.float32)
	label = tf.cast(features['labels'], tf.int32)
	label = tf.reshape(label, [1])

	# image = tf.decode_raw(features['image_raw'], tf.int32)
	# image = tf.cast(image, tf.float32)
	# label = tf.cast(features['labels'], tf.int32)

	# image.set_shape([32, 32])
	# label.set_shape([1])

	return image, label


def generate_image_and_label_batch(image, label, min_after_dequeue, batch_size=50, shuffle=True):
	"""
	capacity must be larger than min_after_dequeue and 
	the amount larger determines the maximum we will prefetch.  
	Recommendation:
	min_after_dequeue + (num_threads + a small safety margin) * batch_size
	"""
	# MIN_AFTER_DEQUEUE = 1000
	CAPACITY = min_after_dequeue + 3 * batch_size
	# image, label = parse_tfrecords('train')
	if shuffle:
		image_batch, label_batch = tf.train.shuffle_batch(
											[image, label], 
											min_after_dequeue=min_after_dequeue,
											batch_size=batch_size, 
											capacity=CAPACITY) #allow smaller final batch参数默认为False 
	else:
		image_batch, label_batch = tf.train.batch(
											[image, label], 
											batch_size=batch_size, 
											capacity=CAPACITY)
	return image_batch, label_batch


#=================================================DUMPED CODES========================================================
def parse_single_tfrecord():
	"""
	for test
	逐条解析example数据，不使用队列
	"""
	for serialized_example in tf.python_io.tf_record_iterator("data/train.tfrecords"):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)

		rows = example.features.feature['height'].int64_list.value
		cols = example.features.feature['width'].int64_list.value
		print(rows[0])
		print(cols[0])

		image_raw = example.features.feature['image_raw'].bytes_list.value
		image = tf.decode_raw(image_raw[0], tf.int32)
		reshaped_img = tf.reshape(image, [rows[0], cols[0]])
		label = example.features.feature['labels'].int64_list.value

		return reshaped_img, image


def load_train_images(idx_ubyte_file=TRAIN_IMAGES_PATH):
	"""
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
	return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file=TRAIN_LABELS_PATH):
	"""
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
	return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file=TEST_IMAGES_PATH):
	"""
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
	return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=TEST_LABELS_PATH):
	"""
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
	return decode_idx1_ubyte(idx_ubyte_file)


class DataSet(object):
	def __init__(self, images, labels):
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._num_examples = images.shape[0]

	@property
	def images(self):
		return _images

	@property
	def labels(self):
		return _labels

	@property
	def num_examples(self):
		return _num_examples

	@property
	def epochs_completed(self):
		return _epochs_completed

	def next_batch(self, batch_size, shuffle=True):
		start = _index_in_epoch
		#为第一个epoch做shuffle
		if(start == 0 and epochs_completed == 0 and shuffle):
			perm0 = np.arange(_num_examples)
			np.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]
		#取出下一个batch的size超出当前epoch的范围时，进入下一个epoch
		if(start + batch_size > self._num_examples):
			self.epochs_completed += 1
			#取出前一个epoch的剩余部分
			num_rest_examples = self._num_examples - start
			images_rest = self._images[start:self._num_examples]
			labels_rest = self._labels[start:self._num_examples]
			#为下一个epoch做shuffle
			if shuffle:
				perm1 = np.arange(self._num_examples)
				np.random.shuffle(perm1)
				self._images = self.images[perm1]
				self._labels = self.labels[perm1]
			start = 0
			_index_in_epoch = batch_size - num_rest_examples
			images_new = self._images[start:_index_in_epoch]
			labels_new = self._labels[start:_index_in_epoch]
			return np.concatenate((images_new, images_rest), axis=0), np.concatenate((labels_new, labels_rest), axis=0)
		#取出的下一个batch的size仍然在当前epoch范围内时
		else:
			self._index_in_epoch += batch_size
			return self._images[start:_index_in_epoch], self._labels[start:_index_in_epoch]


#=======================================================RUN CODE=============================================================
def run():
	train_images = load_train_images()
	train_labels = load_train_labels()

	for i in range(10):
		print('标记' + str(train_labels[i]))
		print('图片' + str(train_images[i]))
		plt.imshow(train_images[i], cmap='gray')
		plt.show()
	print('done')


if __name__ == '__main__':
	generate_tfrecords('test')

	image, label = parse_single_tfrecord()

	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	sess.run(init)

	image_arr = sess.run(image)[:]
	#哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈我干！！原来要从tensor中切片取出来元素组成array-like data	
	print(image_arr)
	print(label)
	plt.imshow(image_arr, cmap='gray')
	plt.show()