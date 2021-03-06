# encoding: utf-8
"""测试对抗训练所得的所有网络模型的精确度，检测率
TEST 1. 测试针对**同种**universal perturbation的效果
TEST 2. 测试针对**不同**universal perturbation 1的效果
TEST 3. 测试针对**不同**universal perturbation 2的效果
"""
import os 
import sys
import argparse

from math import log
from math import sqrt

import tensorflow as tf
import numpy as np

from bottleneck_util import run_bottleneck_on_image, load_npy
from prepare_imagenet_data import preprocess_image_batch

FLAGS = None
BOTTLENECK_TENSOR_SIZE = 1024
OUTPUT_TENSOR_SIZE = 1008
GOOD_SOFTMAX_TENSOR = 'softmax2:0'
BAD_SOFTMAX_TENSOR = 'final_result:0'


def get_saved_labels(path_ground_truth, category, adv_or_not, rows, offset=0):
	"""Retrieve specific number of labels from cache stored in disk.
	"""
	all_labels = load_npy(path_ground_truth, category, adv_or_not)
	res = np.array([all_labels[i][1:] for i in range(rows)]).reshape((-1,))
	if offset > 0:
		res = np.concatenate((body_part, all_labels[rows][1:offset+1]))
	print('>>Retrieve %d labels from cached file.' % res.shape)
	return res


def undo_image_avg(img):
	img_copy = np.copy(img)
	img_copy[:, :, :, 0] = img_copy[:, :, :, 0] + 123.68
	img_copy[:, :, :, 1] = img_copy[:, :, :, 1] + 116.779
	img_copy[:, :, :, 2] = img_copy[:, :, :, 2] + 103.939
	return img_copy


# ===========================================待测试=========================================
def pick_btlnk_label(sess, path_imagenet, pert, how_many, input_tensor, bottleneck_tensor, output_tensor, softmax_tensor, eval_type, T):
	"""
	Pick out valid sample which can be perturbed successfully with the pert.
	***Design for CROSS universal perturbation test***.
	Returns:
	res: Array int containing normal label part and adversarial label part
	bottleneck_lists: Array float64 containing normal bottleneck part 
		and adversarial bottleneck part(前一半adversarial部分，后一半为original部分).
	"""
	half_how_many = int(np.ceil(how_many / 2))
	print('>>PICK PERT: need to pick out %d valid sample...' % half_how_many)

	bottleneck_lists = np.zeros((how_many, BOTTLENECK_TENSOR_SIZE), dtype=np.float)
	if eval_type == 'top':
		res = np.zeros((how_many,), dtype=np.int)
	elif eval_type == 'jsd' or eval_type == 'cos':
		res = np.zeros((how_many, OUTPUT_TENSOR_SIZE), dtype=np.float)

	already_get = 0 # 记录成功采集的对抗样本数
	path_test_set = os.path.join(path_imagenet, 'test')
	filenames = [x[2] for x in os.walk(path_test_set)][0]
	total_num = len(filenames)
	num_of_batch = int(np.ceil(total_num / FLAGS.batch_size))

	for i in range(num_of_batch):
		start = i * FLAGS.batch_size
		end = min((i+1)*FLAGS.batch_size, total_num)
		image_batch = preprocess_image_batch(path_test_set, filenames[start:end], (256,256), (224,224))
		clipped_v = np.clip(undo_image_avg(image_batch + pert), 0, 255) - np.clip(undo_image_avg(image_batch), 0, 255)
		image_perturbed_batch = image_batch + clipped_v

		# 计算一个batch的瓶颈值，输出值
		orin_btlnks = run_bottleneck_on_image(sess, image_batch, 
											input_tensor, bottleneck_tensor)
		adv_btlnks = run_bottleneck_on_image(sess, image_perturbed_batch, 
											input_tensor, bottleneck_tensor)
		orin_logits = run_bottleneck_on_image(sess, orin_btlnks, 
											bottleneck_tensor, output_tensor)
		adv_logits = run_bottleneck_on_image(sess, adv_btlnks, 
											bottleneck_tensor, output_tensor)
		# 挑选出一个batch的干扰成功样本
		orin_labels = np.argmax(orin_logits, axis=1)
		adv_labels = np.argmax(adv_logits, axis=1)
		mask = orin_labels != adv_labels
		valid_num = np.sum(mask)
		temp_already_get = already_get + valid_num

		if temp_already_get >= half_how_many:
			temp_cnt  = half_how_many - already_get

			bottleneck_lists[already_get : half_how_many] = adv_btlnks[mask][:temp_cnt]
			bottleneck_lists[(already_get+half_how_many) : how_many] = orin_btlnks[mask][:temp_cnt]
			if eval_type == 'top':
				res[already_get : half_how_many] = adv_labels[mask][:temp_cnt]
				res[(already_get+half_how_many) : how_many] = orin_labels[mask][:temp_cnt]
			elif eval_type == 'jsd':
				# jsd评估需要的是softmax值，softmax层还能放大激活值的分布差异
				# 不过需要scale参数T防止饱和
				res[already_get : half_how_many] = run_bottleneck_on_image(
														sess, 
														(adv_logits[mask][:temp_cnt] / T), 
														output_tensor, 
														softmax_tensor)
				res[(already_get+half_how_many) : how_many] = run_bottleneck_on_image(
														sess, 
														(orin_logits[mask][:temp_cnt] / T), 
														output_tensor, 
														softmax_tensor)
			elif eval_type == 'cos':
				res[already_get : half_how_many] = adv_logits[mask][:temp_cnt]
				res[(already_get+half_how_many) : how_many] = orin_logits[mask][:temp_cnt]
			else:
				print('++ Warning!! Please choose a evaluation type: top or jsd.')
			print('++ pick_bltnk_label--res shape: ', res.shape)

			break
		else:
			bottleneck_lists[already_get : temp_already_get] = adv_btlnks[mask]
			bottleneck_lists[(already_get+half_how_many) : (temp_already_get+half_how_many)] = orin_btlnks[mask]
			if eval_type == 'top':
				res[already_get : temp_already_get] = adv_labels[mask]
				res[(already_get+half_how_many) : (temp_already_get+half_how_many)] = orin_labels[mask]
			elif eval_type == 'jsd':
				res[already_get : temp_already_get] = run_bottleneck_on_image(
														sess,
														(adv_logits[mask] / T),
														output_tensor, 
														softmax_tensor)
				res[(already_get+half_how_many) : (temp_already_get+half_how_many)] = run_bottleneck_on_image(
														sess,
														(orin_logits[mask] / T),
														output_tensor, 
														softmax_tensor)
			elif eval_type == 'cos':
				res[already_get : temp_already_get] = adv_logits[mask]
				res[(already_get+half_how_many) : (temp_already_get+half_how_many)] = orin_logits[mask]
			else:
				print('++ Warning!! Please choose a evaluation type: top or jsd.')
		already_get = temp_already_get
	print('++ pick res shape: ', res.shape)

	return bottleneck_lists, res


def get_bottleneck_values(btlnk_filenames):
	"""
	Retrieve valid bottleneck values of samples which can be perturbed successfully 
	with the pert.
	***Design for SAME universal perturbation test***.
	Returns:
	Numpy array float of bottleneck values.
	"""
	res = []
	for filename in btlnk_filenames:
		filename = os.path.join(FLAGS.bottleneck_dir, filename)
		with open(filename, 'r') as f:
			btlnk_string = f.read()
		btlnk_values = [float(x) for x in btlnk_string.split(',')]
		res.append(btlnk_values)

	return np.stack(res, axis=0) 


def create_graph(graph_dir, graph_name, return_elements):
	"""Create a graph from saved GraphDef file and returns a Graph object.
	Args:
	return_elements: List String with limit of 3 elements.
	"""
	print('>>Creating graph %s from saved GraphDef file...' % graph_name)
	path_graph = os.path.join(graph_dir, graph_name)

	with tf.Session() as sess:
		with tf.gfile.FastGFile(path_graph, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			
	return sess.graph, tf.import_graph_def(graph_def, name='', return_elements=return_elements)


# ============================= JSD EVALUATION PART ================================
def KLD(p, q):
	return sum([_p * log(_p,2) - _p * log(_q,2) for (_p, _q) in zip(p,q)])


def JSD_core(p, q):
	mask = np.ones((p.shape[0],), dtype=np.int)
	# p, q = zip(filter(lambda x : x[0]!=0 or x[1]!=0, [(_p, _q) for _p,_q in zip(p,q)])) # 去掉p，q中同时为0的元素
	i = 0
	for _p, _q in zip(p, q):
		if _p == 0 and _q == 0:
			mask[i] = 0
		i += 1
	p = p[mask]
	q = q[mask] 

	M = [0.5 * (_p + _q) for _p,_q in zip(p,q)]
	p += np.spacing(1)
	q += np.spacing(1)
	M += np.spacing(1)

	return 0.5 * KLD(p, M) + 0.5 * KLD(q, M)


def Cosine(vec1, vec2):
	# 用余弦夹角衡量样本间差异
    return vec1.dot(vec2)/(sqrt((vec1**2).sum()) * sqrt((vec2**2).sum()))


# ============================= MAIN PART ================================
def main(_):
	# ============================= GOOD BRO PART ================================
	if FLAGS.dataset_size % 2 != 0:
		raise Exception('Error: parameter dataset_size should be a even number!')
	if FLAGS.pert_path == '' and FLAGS.test_type == 'CROSS':
		raise Exception('Warning: please specify a perturbation path!') 
	if FLAGS.adv_graph_name == '':
		raise Exception('Warning: please specify a bad bro model graph path!') 

	half_dataset_size = int(FLAGS.dataset_size / 2)

	if FLAGS.test_type == 'SAME':
		# for SAME perturbation test:
		# retrieve bottlenecks and labels from cache stored in disk directly
		# summary（normal：5000 + adversarial:5000）
		rows = int(half_dataset_size / 50)
		offset = half_dataset_size % 50

		adv_labels = get_saved_labels(FLAGS.ground_truth_dir, 'test', True, rows, offset) 
		norm_labels = get_saved_labels(FLAGS.ground_truth_dir, 'test', False, rows, offset) 
		test_good_labels = np.concatenate((adv_labels, norm_labels))
		print('good labels shape: ', test_good_labels.shape)

		# 同干扰测试：bad bro输入瓶颈值（normal：5000 + adversarial:5000）
		bottleneck_lists = [x[2] for x in os.walk(FLAGS.bottleneck_dir)][0]
		half_btlnk_size = int(len(bottleneck_lists) / 2)
		btlnk_filenames = np.concatenate((bottleneck_lists[0 : half_dataset_size], 
											bottleneck_lists[half_btlnk_size : half_btlnk_size+half_dataset_size]))
		print('>>Get %d bottleneck filenames.' % btlnk_filenames.shape)
		test_bottlenecks = get_bottleneck_values(btlnk_filenames)

	elif FLAGS.test_type == 'CROSS':
		# for different perturbation cross test:
		pert = np.load(FLAGS.pert_path)
		# for CROSS perturbation test: 
		# load the graph , pick out valid bottlenecks and labels from scratch. 
		norm_graph, (norm_input, norm_bottleneck, norm_output, norm_softmax) = create_graph(
					FLAGS.graph_dir, 
					FLAGS.norm_graph_name, 
					[FLAGS.input_tensor, FLAGS.bottleneck_tensor, FLAGS.norm_output_tensor, GOOD_SOFTMAX_TENSOR])

		with tf.Session() as sess:
			test_bottlenecks, test_good_labels = pick_btlnk_label(sess, FLAGS.imagenet_dir, pert, FLAGS.dataset_size, 
						norm_input, norm_bottleneck, norm_output, norm_softmax, FLAGS.eval_type, FLAGS.T)


	# ============================= BAD BRO PART ================================
	with tf.Graph().as_default():
		# test_bad_labels = np.zeros((FLAGS.dataset_size,), dtype=np.int)
		if FLAGS.eval_type == 'top':
			test_bad_labels_2 = np.zeros((FLAGS.dataset_size, FLAGS.magic_num), dtype=np.int)
		elif FLAGS.eval_type == 'jsd' or FLAGS.eval_type == 'cos':
			test_bad_logits_2 = np.zeros((FLAGS.dataset_size, OUTPUT_TENSOR_SIZE), dtype=np.float)

		# load bad bro model graph
		adv_graph, (adv_bottleneck, adv_output, adv_softmax) = create_graph(
					FLAGS.graph_dir, 
					FLAGS.adv_graph_name, 
					[FLAGS.bottleneck_tensor, FLAGS.adv_output_tensor, BAD_SOFTMAX_TENSOR])
		num_batches = np.int(np.ceil(FLAGS.dataset_size / FLAGS.batch_size))

		with tf.Session() as sess:
			for i in range(num_batches):
				start = i * FLAGS.batch_size
				end = min((i+1)*FLAGS.batch_size, FLAGS.dataset_size)
				logits = run_bottleneck_on_image(sess, test_bottlenecks[start:end], 
													adv_bottleneck, adv_output)
				if FLAGS.eval_type == 'jsd':
					test_bad_logits_2[start:end] = run_bottleneck_on_image(
														sess, 
														(logits/FLAGS.T),
														adv_output,
														adv_softmax) 
				elif FLAGS.eval_type == 'cos':
					test_bad_logits_2[start:end] = logits
				elif FLAGS.eval_type == 'top':
					# test_bad_labels[start:end] = np.argmax(logits, axis=1)
					test_bad_labels_2[start:end] = np.argsort(-logits, axis=1)[:, :FLAGS.magic_num]


	# ============================= FUSE INFO PART ================================
	# adv_result = np.sum(test_good_labels[:half_dataset_size] == test_bad_labels[:half_dataset_size])
	# norm_result = np.sum(test_good_labels[half_dataset_size:] != test_bad_labels[:half_dataset_size])
	if FLAGS.eval_type == 'top':
		# adversarial part
		adv_cnt = 0
		for i in range(half_dataset_size):
			if test_good_labels[:half_dataset_size][i] in test_bad_labels_2[:half_dataset_size][i]:
				adv_cnt += 1
		adv_accuracy = float(adv_cnt) / float(half_dataset_size)

		# normal part
		norm_cnt = 0
		for i in range(half_dataset_size):
			if test_good_labels[half_dataset_size:][i] in test_bad_labels_2[half_dataset_size:][i]:
				norm_cnt += 1
		adv_accuracy = float(adv_cnt) / float(half_dataset_size)
		norm_accuracy = float(half_dataset_size - norm_cnt) / float(half_dataset_size)
		total_accuracy = float(adv_cnt + half_dataset_size - norm_cnt) / float(FLAGS.dataset_size)

		# total_accuracy = float(adv_result + norm_result) / float(FLAGS.dataset_size)
		# adv_accuracy = float(adv_result) / float(half_dataset_size)
		# norm_accuracy = float(norm_result) / float(half_dataset_size)

		print('|++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++|')
		print('|++++++++++++++++++++++++ TEST  RESULT ++++++++++++++++++++++++++|')
		print('|++ TYPE: %s BAD BRO MODEL: %s ++|' % (FLAGS.test_type, FLAGS.adv_graph_name))
		print('|++ TOTAL ACCURACY: %d%%, ADV ACCURACY: %d%%, NORM ACCURACY: %d%% ++|' % 
					(total_accuracy*100, adv_accuracy*100, norm_accuracy*100))
		print('|++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++|')
	elif FLAGS.eval_type == 'jsd':
		jsd_list = np.zeros((FLAGS.dataset_size,), dtype=np.float)
		# JS Divergence
		for i in range(FLAGS.dataset_size):
			# 历史遗留问题，eval_type为jsd时，test_good_lables保存的是倒数第二层激活值向量
			jsd_val = JSD_core(test_good_labels[i], test_bad_logits_2[i])
			# print('jsd value: ', jsd_val)
			jsd_list[i] = jsd_val
		# print(jsd_list)
		np.save('data/pca/jsd_val.npy', jsd_list)
	elif FLAGS.eval_type == 'cos':
		cos_list = np.zeros((FLAGS.dataset_size,), dtype=np.float)
		for i in range(FLAGS.dataset_size):
			cos_val = Cosine(test_good_labels[i], test_bad_logits_2[i])
			cos_list[i] = cos_val
		np.save('data/pca/cos_val.npy', cos_list)
	else:
		print('Warning!! FUSE INFO PART -- Please choose a evaluation type: top or jsd.')


# ============================= MAIN CODE PART ================================
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--graph_dir',
			type=str,
			default='data/graph',
			help='path to pretrained model file.')
	parser.add_argument(
			'--bottleneck_dir',
			type=str,
			default='data/bottleneck/test',
			help='path to cached test bottleneck file.')
	parser.add_argument(
			'--ground_truth_dir',
			type=str,
			default='data/ground_truth',
			help='path to cached test label file.')
	parser.add_argument(
			'--imagenet_dir',
			type=str,
			default='D:/Scholarship/dataset/ILSVRC2012',
			help='path to imagenet dataset.')
	parser.add_argument(
			'--adv_output_tensor',
			type=str,
			default='final_training_ops/Wx_plus_b/add:0',
			help='name of adversarial model\'s output tensor.')
	parser.add_argument(
			'--norm_output_tensor',
			type=str,
			default='softmax2_pre_activation:0',
			help='name of normal model\'s output tensor.')
	parser.add_argument(
			'--bottleneck_tensor',
			type=str,
			default='avgpool0/reshape:0',
			help='name of model\'s bottleneck tensor.')
	parser.add_argument(
			'--input_tensor',
			type=str,
			default='input:0',
			help='name of model\'s input tensor.')
	parser.add_argument(
			'--norm_graph_name',
			type=str,
			default='tensorflow_inception_graph.pb',
			help='name of normal model graph.')
	parser.add_argument(
			'--adv_graph_name',
			type=str,
			default='',
			help='name of adversarial model graph.')
	parser.add_argument(
			'--pert_path',
			type=str,
			default='',
			help='path to universal adversarial perturbation.')
	parser.add_argument(
			'--test_type',
			type=str,
			default='SAME',
			help='type of test(SAME for same perturbation, CROSS for different perturbation).')
	parser.add_argument(
			'--eval_type',
			type=str,
			default='top',
			help='type of framework evaluation(top or jsd or cos).')
	parser.add_argument(
			'--batch_size',
			type=int,
			default=100,
			help='size of testing batch.') 
	parser.add_argument(
			'--dataset_size',
			type=int,
			default=10000,
			help='size of total testing set, must be even.')
	parser.add_argument(
			'--magic_num',
			type=int,
			default=1,
			help='')
	parser.add_argument(
			'--T',
			type=float,
			default=1.0,
			help='softmax layer scale item, in case of saturating.')

	FLAGS, unparsed = parser.parse_known_args()

	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)