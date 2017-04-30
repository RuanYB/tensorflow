import tensorflow as tf
import numpy as np
import mnist

def dense_to_one_hot(label_dense, num_classes):
	"""
	将表示类别的label标量转换为one-hot向量
	Args:
	label_dense： 标量形式的label数组
	num_classes: 相互独立的分类结果的数量
	"""
	num_labels = label_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + label_dense.ravel()] = 1
	return labels_one_hot


def train():
	"""
	训练mnist网络
	"""
	

def test():
	label_dense = np.array([[2], [3], [4]])
	result = dense_to_one_hot(label_dense, 10)
	print(result)

if __name__ == '__main__':
	test()