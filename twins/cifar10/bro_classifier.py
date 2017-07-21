# encoding: utf-8
"""
使用brother模型检测对抗样本
验证top-1，top-3，top-5的实验结果
"""
import os
import sys
import argparse

import tensorflow as tf 
import numpy as np

from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf

from cifar10_input import generate_data
from demo_cifar import pick_pert


INPUT_TENSOR_NAME = 'input:0'
ADV_SOFTMAX_TENSOR_NAME = 'final_result:0'
ORIN_ACTIVATION_NAME = 'softmax_linear/softmax_linear:0'
ADV_ACTIVATION_NAME = 'final_training_ops/final_linear/Wx_plus_b/add:0'
BOTTLENECK_TENSOR_NAME = 'local3/Reshape:0'
BOTTLENECK_TENSOR_SIZE = 2304


def load_npy(file_type, file_name, npy_dir):
	"""
	Helper function for loading training of testing .npy files.
	Args:
		file_type: type of train or test.
		file_name: file name of image list or label list.
	Returns:
		numpy array of cached .npy file.
	"""
	path = os.path.join(npy_dir, ('%s_%s.npy' % (file_type, file_name)))
	return np.load(path)


def get_bottleneck_value(start, end):
	"""根据预计算的test集bottleneck文件路径获取bottleneck值并返回
	"""
	# print('>>Retrieving Bottleneck Value of Sample %d to %d...' % (start, end))	

	bottleneck_path = [os.path.join(FLAGS.bottleneck_dir, 'test', 'test_%d.txt' % index) for index in range(start, end)]

	result = np.zeros([(end-start), BOTTLENECK_TENSOR_SIZE], dtype=np.float32)
	index = 0

	for path in bottleneck_path:
		with open(str(path), 'r') as bottleneck_file:
			bottleneck_string = bottleneck_file.read()
		try:
			bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
		except:
			print(">>GET BOTTLENECK VALUE: Invalid float found, recreating bottleneck!!")

		result[index] = bottleneck_values
		index += 1

	return result


def bottleneck_eval(step_size, data_size, ff):
	"""根据bottleneck值计算最终的激活值lsit
	"""
	num_evals = int(np.ceil(float(data_size) / float(step_size)))
	result_list = np.zeros([data_size, 10], dtype=np.float32)

	for i in range(num_evals):
		start = step_size * i
		end = min(step_size*(1+i), data_size)
		bottleneck_values = get_bottleneck_value(start, end)

		result_list[start:end] = ff(np.reshape(bottleneck_values, (-1, BOTTLENECK_TENSOR_SIZE)))

	return result_list


def input_eval(step_size, dataset, ff):
	"""根据input image值计算最终的激活值lsit
	"""
	data_size = dataset.shape[0]

	num_evals = int(np.ceil(float(data_size) / float(step_size)))
	result_list = np.zeros([data_size, 10], dtype=np.float32)

	for i in range(num_evals):
		start = step_size * i
		end = min(step_size*(1+i), data_size)
		input_values = dataset[start:end, :, :, :]

		result_list[start:end] = ff(np.reshape(input_values, (-1, 24, 24, 3)))

	return result_list


# def top_n_accuray(orin_labels, adv_labels, data_size, top_n):
# 	"""
# 	Args:
# 		orin_labels: labels predicted by original classifier of shape [datasize]
# 		adv_labels: labels predicted by adversarial classifier of shape [datasize, 5]
# 	Returns:
# 		accuracy: top-n accuracy
# 	"""
# 	error_sum = 0

# 	for i in range(top_n):
# 		# error_sum += np.sum(orin_labels.flatten() == adv_labels[:, i:(i+1)].flatten())
# 		# mask = (orin_labels.flatten() == adv_labels[:, i:(i+1)].flatten())

# 		mask = (orin_labels.flatten() == adv_labels.flatten())
# 		error_sum += np.sum(mask)
# 		mask_orin_labels = orin_labels.flatten()[mask]
# 		# mask_adv_labels = adv_labels.flatten[mask]
# 		#统计每个误识别的样本的label数量
# 		for i in range(10):
# 			print('label %d num: %d' % (i, np.sum(mask_orin_labels == i)))

# 	accuracy = float(error_sum) / float(data_size)

# 	return accuracy


# ===================== MAIN CODE =====================
def main(_):
	# prepare test set
	test_bottleneck_dir = os.path.join(FLAGS.bottleneck_dir, 'test')

	sum_path = os.path.join(FLAGS.pert_dir, 'pert_sum.txt')
	# Load perturbated image set size info of test set.
	with open(sum_path, 'r') as pert_sum_file:
		pert_sum_string = pert_sum_file.read()
	pert_sum_values = [x for x in pert_sum_string.split(',')]
	test_set_size = int(pert_sum_values[2])

	# 加载good bro model
	gdef = gpb.GraphDef()
	model_file_name = os.path.join(FLAGS.model_dir, 'cifar_freeze_graph.pbtxt') 
	with open(model_file_name, 'r') as fh:
		graph_str = fh.read()

	pbtf.Parse(graph_str, gdef)

	orin_bottleneck_tensor, orin_activation_tensor, orin_input_tensor = tf.import_graph_def(
			gdef, 
			name='', 
			return_elements=[BOTTLENECK_TENSOR_NAME, 
					ORIN_ACTIVATION_NAME,
					INPUT_TENSOR_NAME])

	with tf.Session() as sess:
		def orin_bottleneck_forward(inp):
			return sess.run(orin_activation_tensor, feed_dict={
					orin_bottleneck_tensor: np.reshape(inp, (-1, BOTTLENECK_TENSOR_SIZE))})

		# 同种干扰向量测试：对抗样本经过good bro之后的输出
		orin_list = bottleneck_eval(FLAGS.eval_interval, test_set_size, orin_bottleneck_forward)

		def orin_input_forward(inp):
			return sess.run(orin_activation_tensor, feed_dict={
					orin_input_tensor: np.reshape(inp, (-1, 24, 24, 3))})

		# 重新加载测试集
		_, testing_set = generate_data(sess, 'data/cifar-10-batches-bin', training_size=0, testing_size=10000)
		v = np.load(os.path.join(FLAGS.pert_dir, 'universal_for_test.npy'))
		# 挑选能够成功干扰的测试样本
		testing_list = pick_pert(sess, v, orin_bottleneck_forward, testing_set, 'test', save=False)


	with tf.Graph().as_default():
		# 加载bad bro model
		with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, FLAGS.pb_dir), 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())

			adv_bottleneck_tensor, adv_activation_tensor, adv_input_tensor = tf.import_graph_def(graph_def, name='', return_elements=[
								BOTTLENECK_TENSOR_NAME, 
								ADV_ACTIVATION_NAME,
								INPUT_TENSOR_NAME])
			with tf.Session() as sess:
				def adv_bottleneck_forward(inp):
					"""输入tensor为瓶颈tensor：local3_reshape
					"""
					return sess.run(adv_activation_tensor, feed_dict={
							adv_bottleneck_tensor: inp})
				# 同种干扰向量测试：对抗样本经过bad bro之后的输出
				adv_list = bottleneck_eval(FLAGS.eval_interval, test_set_size, adv_bottleneck_forward)

				def adv_input_forward(inp):
					"""输入tensor为起始tensor：input
					"""
					return sess.run(adv_activation_tensor, feed_dict={
							adv_input_tensor: inp})
				# 不同干扰向量测试：对抗样本经过bad bro之后的输出
				normal_result_list = input_eval(step_size=FLAGS.eval_interval, 
											dataset=testing_list['normal_image'], 
											ff=adv_input_forward)
				perted_result_list = input_eval(step_size=FLAGS.eval_interval, 
											dataset=testing_list['perted_image'], 
											ff=adv_input_forward)

	# 同种干扰向量测试：good bro和bad bro的输出label
	orin_labels = np.argmax(orin_list, 1)
	adv_labels = np.argmax(adv_list, 1)
	
	# adv_labels = np.argsort(-adv_list, axis=1)
	# adv_labels = adv_labels[:, 0:5]
	# test_fake_labels = load_npy('test', 'fake_label', FLAGS.pert_dir).astype(int)

	half_test_size = int(test_set_size // 2)
	# 同种干扰向量测试：恶意样本识别正常的数量
	sum_adv = np.sum(orin_labels.flatten()[:half_test_size] == adv_labels.flatten()[:half_test_size])
	# 同种干扰向量测试：正常样本识别正常的数量
	sum_norm = np.sum(orin_labels.flatten()[half_test_size:] != adv_labels.flatten()[half_test_size:])

	# top-1准确度(综合)
	# 由于test bottleneck集合前一半为恶意样本，后一半为正常样本
	# 所以正确率取前半部分相等的和后半部分不等的数量
	accuracy_top_1 = float(sum_adv + sum_norm) / float(test_set_size)
	
	# False Positive结果
	# 恶意样本被认为正常的比率，基数为测试集一半大小
	fp_top_1 = 1.0 - float(sum_adv / half_test_size)

	# fn_norm_labels = np.argmax(fn_orin_list, axis=1)
	# fn_adv_labels = np.argmax(fn_adv_list, axis=1)

	# 正常样本被认为恶意的比率（false alarm rate）
	fn_top_1 = 1.0 - float(sum_norm / half_test_size)

	fmt = '|{0:^8}|{1:^20.4f}|{2:^21.4f}|'

	print('|+++++++++++++++++++++++++++++++++++++++++++++++++++|')
	print('|            BAD BRO Detection Statistics           |')
	print('|  NORM PART SIZE:%5d   |   ADV PART SIZE:%5d   |' % (half_test_size, half_test_size))
	print('|           Total Detection Rate: %6.4f            |' % accuracy_top_1)
	print('|  TOPn | False Positive Rate | False Negative Rate |')
	print(fmt.format(1, fp_top_1, fn_top_1))
	print('|+++++++++++++++++++++++++++++++++++++++++++++++++++|')
	print('说明：\r\n1. Total Detection Rate基数为整体样本集\r\n2. FP和FN Rate基数为半数样本集')




# ===================== LAUNCH CODE =====================
if __name__  == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--model_dir', 
			type=str, 
			default='pb', 
			help='Path to cifar_freeze_def.pb')
	parser.add_argument(
			'--bottleneck_dir', 
			type=str, 
			default='data/bottleneck', 
			help='Path to pre-processed bottleneck files.')
	parser.add_argument(
			'--pert_dir', 
			type=str, 
			default='data/pert', 
			help='Path to perturbation files.')
	parser.add_argument(
			'--bins_dir', 
			type=str, 
			default='data/cifar-10-batches-bin', 
			help='Path to .bin files.')
	parser.add_argument(
			'--pb_dir', 
			type=str, 
			default='pb/adv_graph_norm_as_fake_2w.pb', 
			help='Path to pre-trained .pb files.')
	parser.add_argument(
			'--eval_interval', 
			type=int, 
			default=100, 
			help='How often to evaluate the test example.')

	FLAGS, unparsed= parser.parse_known_args()

	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)