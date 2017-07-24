# encoding: utf-8
"""
transfer learning with a simple model for cifar10 learning.
"""
import os
import sys
import argparse
import random
from datetime import datetime

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf

FLAGS = None

# inception网络中全连接层只有2层，该模型有3层，这里暂且取倒数第三层为bottle neck
BOTTLENECK_TENSOR_NAME = 'local3/Reshape:0'
BOTTLENECK_TENSOR_SIZE = 2304
MODEL_INPUT_HEIGHT = 24
MODEL_INPUT_WIDTH = 24
MODEL_INPUT_DEPTH = 3 
INPUT_TENSOR_NAME = 'input:0'
SOFTMAX_TENSOR = 'softmax_linear/softmax_linear:0'


def ensure_dir_exists(dir_name):
	"""
	Make sure the folder exists on disk.
	"""
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)


def variable_summaries(var):
	"""
	Attach a lot of summaries to a Tensor for Tensorboard visualization.
	"""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


def get_bottleneck_path(bottleneck_type, index, index_bound, bottleneck_dir):
	"""
	Returns a path to a cached bottleneck file for a given label.
	Args:
		bottleneck_type: type of train or test or validate.
		index: index of bottleneck file which is about to retrieve.
		index_bound: upper bound of index of given bottleneck type.
		bottleneck_dir: path Where to retrieve pre-processed bottleneck files.
	Returns:
		bottleneck values: numpy array of the same shape as bottleneck tensor.
	"""
	if index < 0 or index >= index_bound:
		raise KeyError('>>GET BOTTLEPATH: Invalid index: %d' % index)

	file_name = '%s_%d.txt' % (bottleneck_type, index)
	return os.path.join(bottleneck_dir, bottleneck_type, file_name)


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


def special_random_num(except_num, start=0, end=1):
	"""generate a random number between start and end except for specific number.
	"""
	temp = except_num
	while  temp == except_num:
		temp = random.randrange(start, end)
	# print('formal num:%d, latter num:%d' % (except_num, temp))
	return temp


def create_cifar_graph():
	"""
	Creates a graph from saved GraphDef file and returns a Graph object.
	"""
	# ckpt = tf.train.get_checkpoint_state('cifar10_train')
	# model_path = ckpt.model_checkpoint_path
	# print('>>模型存储路径： ', model_path)
	# # 元数据meta路径
	# model_meta_path = '.'.join([model_path, 'meta'])

	# new_saver = tf.train.import_meta_graph(model_meta_path, clear_devices=True)

	# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		# new_saver.restore(sess, model_path)
		# graph = tf.get_default_graph()

		# input_tensor = graph.get_tensor_by_name(INPUT_TENSOR_NAME)
		# bottleneck_tensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)


	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		gdef = gpb.GraphDef()
		model_file_name = os.path.join(FLAGS.model_dir, 'cifar_freeze_graph.pbtxt') 
		with open(model_file_name, 'r') as fh:
			graph_str = fh.read()

		pbtf.Parse(graph_str, gdef)

		bottleneck_tensor, input_tensor = tf.import_graph_def(gdef, name='', return_elements=[
									BOTTLENECK_TENSOR_NAME, 
									INPUT_TENSOR_NAME])
	
		# with gfile.FastGFile('data/cifar_freeze_graph.pb', 'rb') as f:
		# 	graph_def = tf.GraphDef()
		# 	graph_def.ParseFromString(f.read())
		# 	# sess.graph.as_default()
		# 	bottleneck_tensor, input_tensor = tf.import_graph_def(graph_def, 
		# 				name='', return_elements=[
		# 							BOTTLENECK_TENSOR_NAME, 
		# 							INPUT_TENSOR_NAME])
	return sess.graph, bottleneck_tensor, input_tensor



def get_bottleneck_value(bottleneck_path):
	"""
	Retrieve cached bottleneck values for an image of specific path.
	"""
	with open(bottleneck_path, 'r') as bottleneck_file:
		bottleneck_string = bottleneck_file.read()
	try:
		bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
	except:
		print(">>GET BOTTLENECK VALUE: Invalid float found, recreating bottleneck!!")

	return np.array(bottleneck_values)


def get_random_cached_bottlenecks(how_many, bottleneck_type, index_bound, label_list):
	"""
	Retrieve random bottleneck values for cached images.
	Add no distortion in consideration of invalidation of adversarial perturbation been distorted.
	Args:
		how_many: number of bottlenecks to retrieve.
		index_bound: upper bound of specific type of bottleneck.
		label_list: corresponding ground truths of specific type.
		bottleneck_type: trian or test or validate.
	Returns:
		List of bottleneck arrays, their corresponding ground truths, and the
		relevant filenames.
	"""
	bottlenecks = []
	ground_truths = []
	filenames = []
	i = 0
	half_bound = index_bound // 2

	if how_many > 0:
		# Retrieve a random sample of bottlenecks.
		while i < how_many:
			index = random.randrange(index_bound)
			path = get_bottleneck_path(bottleneck_type, index, index_bound, FLAGS.bottleneck_dir)

			bottleneck = get_bottleneck_value(path)

			# ==================================================
			# option 1: set normal example's label as corresponding adversarial example's label
			# ground_truth = label_list[int(index % half_bound)]

			# (unfeasible)option 2: set normal example's label as a out-of-bound value, e.g. 10
			# if index >= half_bound:
			# 	ground_truth = 10.0
			# else:
			# 	ground_truth = int(label_list[index])

			# option 3: set normal example's labels as a random number except for original label
			if index < half_bound:
				ground_truth = int(label_list[index])
			else:
				ground_truth = special_random_num(int(label_list[index-half_bound]), end=9)
			# ==================================================

			bottlenecks.append(bottleneck)
			ground_truths.append(ground_truth)
			filenames.append(path)

			i += 1
	else:
		# Retrieve all bottlenecks
		while i < index_bound:
			path = get_bottleneck_path(bottleneck_type, i, index_bound, FLAGS.bottleneck_dir)

			bottleneck = get_bottleneck_value(path)

			# ==================================================
			# option 1: set normal example's label as corresponding adversarial example's label
			# ground_truth = label_list[int(i % half_bound)]

			# (unfeasible)option 2: set normal example's label as a out-of-bound value, e.g. 10
			# if i >= half_bound:
			# 	ground_truth = 10
			# else:
			# 	ground_truth = int(label_list[index])

			# option 3: set normal example's labels as a random number except for original label
			if i < half_bound:
				ground_truth = int(label_list[i])
			else:
				ground_truth = special_random_num(int(label_list[i-half_bound]), end=9)
			# ==================================================

			bottlenecks.append(bottleneck)
			ground_truths.append(ground_truth)
			filenames.append(path)

			i += 1

	return bottlenecks, ground_truths, filenames


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
	"""
	Add a new softmax and two fully-connected layer for training.
	"""
	with tf.name_scope('input'):
		bottleneck_input = tf.placeholder_with_default(
			bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
			name='BottleneckInputPlaceholder')
		ground_truth_input = tf.placeholder(tf.int32,
											[None],
											name='GroundTruthInput')

	with tf.name_scope('final_training_ops'):
		with tf.name_scope('local3'):
			with tf.name_scope('weights'):
				local3_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 384], stddev=0.001), name='local3_weights')
				variable_summaries(local3_weights)
			with tf.name_scope('biases'):
				local3_biases = tf.Variable(tf.zeros([384]), name='local3_biases')
				variable_summaries(local3_biases)
			with tf.name_scope('Wx_plus_b'):
				local3 = tf.matmul(bottleneck_input, local3_weights) + local3_biases
				tf.summary.histogram('local3_activations', local3)

		with tf.name_scope('local4'):
			with tf.name_scope('weights'):
				local4_weights = tf.Variable(tf.truncated_normal([384, 192], stddev=0.001), name='local4_weights')
				variable_summaries(local4_weights)
			with tf.name_scope('biases'):
				local4_biases = tf.Variable(tf.zeros([192]), name='local4_biases')
				variable_summaries(local4_biases)
			with tf.name_scope('Wx_plus_b'):
				local4 = tf.matmul(local3, local4_weights) + local4_biases
				tf.summary.histogram('local4_activations', local4)

		with tf.name_scope('final_linear'):
			with tf.name_scope('weights'):
				final_weights = tf.Variable(tf.truncated_normal([192, class_count], stddev=0.001), name='final_weights')
				variable_summaries(final_weights)
			with tf.name_scope('biases'):
				final_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
				variable_summaries(final_biases)
			with tf.name_scope('Wx_plus_b'):
				logits = tf.matmul(local4, final_weights) + final_biases
				tf.summary.histogram('pre_activations', logits)

	final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
	tf.summary.histogram('activations', final_tensor)

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=ground_truth_input, 
				logits=logits)
		with tf.name_scope('total'):
			cross_entropy_mean = tf.reduce_mean(cross_entropy)
	tf.summary.scalar('cross_entropy', cross_entropy_mean)

	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
				cross_entropy_mean)

	return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor


def add_evaluation_step(result_tensor, ground_truth_tensor):
	"""
	Insert the operations we need to evaluate the accuracy of our results.
	Args:
		result_tensor: The new final node that produces results.
		ground_truth_tensor: The node we feed ground truth data into.
	Returns:
		Tuple of (evaluation step, prediction).
	"""
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			prediction = tf.argmax(result_tensor, 1)
			correct_prediction = tf.equal(
				prediction, tf.cast(ground_truth_tensor, tf.int64))
		with tf.name_scope('accuracy'):
			evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', evaluation_step)

	return evaluation_step, prediction


def main(_):
	sum_path = os.path.join(FLAGS.pert_dir, 'pert_sum.txt')
	# Load perturbated image set size info of train, test n validate.
	with open(sum_path, 'r') as pert_sum_file:
		pert_sum_string = pert_sum_file.read()
	pert_sum_values = [x for x in pert_sum_string.split(',')]
	train_set_size = int(pert_sum_values[0])
	test_set_size = int(pert_sum_values[2])
	validate_set_size = int(pert_sum_values[3])
	print('>>BOTTLENECK INFO: Train size:%d, Test size:%d, Validate size:%d' 
			% (train_set_size, test_set_size, validate_set_size))

	# load saved fake label array from .npy file
	print('>>MAIN INFO: Start to load label array from .npy file...')
	train_fake_labels = load_npy('train', 'fake_label', FLAGS.npy_dir)
	test_fake_labels = load_npy('test', 'fake_label', FLAGS.npy_dir)
	validate_fake_labels = test_fake_labels[(test_set_size // 2):]
	test_fake_labels = test_fake_labels[:(test_set_size // 2)]

	print('>>MAIN INFO: Start to create new graph...')
	graph, bottleneck_tensor, input_tensor = create_cifar_graph()

	# Add the new layer that we'll be training.
	train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor = add_final_training_ops(
			10, FLAGS.final_tensor_name, bottleneck_tensor)

	# Create the operations we need to evaluate the accuracy of our new layer.
	evaluation_step, prediction = add_evaluation_step(final_tensor, 
													ground_truth_input)

	sess = tf.Session()

	# Merge all the summaries and write them out to /tmp/retrain_logs (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
	validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

	# Set up all our weights to their initial default values.
	init = tf.global_variables_initializer()
	sess.run(init)

	# Run the training for as many cycles as requested on the command line.
	for i in range(FLAGS.how_many_training_steps):
		train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
															FLAGS.train_batch_size, 
															'train', 
															train_set_size-1, 
															train_fake_labels)

		train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: np.array(train_bottlenecks),
																	ground_truth_input: np.array(train_ground_truth)})		

		train_writer.add_summary(train_summary, i)

		# Every so often, print out how well the graph is training.
		is_last_step = (i + 1 == FLAGS.how_many_training_steps)
		if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
			train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={
					bottleneck_input: np.array(train_bottlenecks),
					ground_truth_input: np.array(train_ground_truth)})
			print('|+++++++++++++++++++++++++++++++++++++++++++++++')
			print('|++EVAL: %s: Step %d: Train accuracy = %.1f%% @ Cross entropy = %f' % 
					(datetime.now(), i, train_accuracy * 100, cross_entropy_value))
			
			validation_bottlenecks, validation_ground_truth, _ = get_random_cached_bottlenecks(
																	FLAGS.validation_batch_size,
																	'validate',
																	validate_set_size-1,
																	validate_fake_labels)

			# Run a validation step and capture training summaries for TensorBoard
			# with the `merged` op.
			validation_summary, validation_accuracy = sess.run([merged, evaluation_step], 
						feed_dict={
							bottleneck_input: validation_bottlenecks, 
							ground_truth_input: validation_ground_truth})
			validation_writer.add_summary(validation_summary, i)
			print('|++EVAL: %s: Step %d: Validation accuracy = %.1f%% (N=%d)' % 
					(datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))
		if i % 100 == 0:
			print('|++RETRAIN STEP:', i)


	# completed all the training, so run a final test evaluation on
	# some new images we haven't used before.
	test_bottlenecks, test_ground_truth, test_filenames = get_random_cached_bottlenecks(
																FLAGS.test_batch_size,
																'test',
																test_set_size-1,
																test_fake_labels)
	test_accuracy, predictions = sess.run([evaluation_step, prediction], 
											feed_dict={
												bottleneck_input: test_bottlenecks, 
												ground_truth_input: test_ground_truth})
	print('Final test accuracy = %.1f%% (N=%d)' % 
			(test_accuracy * 100, len(test_bottlenecks)))

	# if print misclassified test images or not
	if FLAGS.print_misclassified_test_images:
		print('====== MISCLASSIFIED TEST IMAGES LIST ======')
		for i, test_filename in enumerate(test_filenames):
			if predictions[i] != test_ground_truth[i]:
				print('%20s :: %s' % (test_filename, predictions[i]))

	# Write out the trained graph and labels with the weights stored as constants.
	output_graph_def = graph_util.convert_variables_to_constants(
			sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
	with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
		f.write(output_graph_def.SerializeToString())
	# with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
	# 	f.write('\n'.join(image_lists.keys()) + '\n')


# ======================= LAUNCH CODE =======================
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
			'--model_dir', 
			type=str, 
			default='pb', 
			help='Path to cifar_freeze_def.pb')
	parser.add_argument(
			'--bins_dir', 
			type=str, 
			default='data/cifar-10-batches-bin', 
			help='Path to cifar10 testing, validation n testing data.')
	parser.add_argument(
			'--pert_dir', 
			type=str, 
			default='D:/Workspace/tensorflow/twins/cifar10/data/pert', 
			help='Directory where to restore or retrieve data.')
	parser.add_argument(
			'--final_tensor_name',
			type=str,
			default='final_result',
			help='\
			The name of the output classification layer in the retrained graph.')
	parser.add_argument(
			'--summaries_dir',
			type=str,
			default='data/retrain_logs',
			help='Where to save summary logs for TensorBoard.')
	parser.add_argument(
			'--npy_dir',
			type=str,
			default='data/pert',
			help='Where to save .npy files for perturbated examples.')
	parser.add_argument(
			'--bottleneck_dir',
			type=str,
			default='data/bottleneck',
			help='Where to retrieve pre-processed bottleneck files.')
	parser.add_argument(
			'--output_graph',
			type=str,
			default='data/retrain_logs/retrain_output_graph.pb',
			help='Where to save the trained graph.')
	parser.add_argument(
			'--train_batch_size',
			type=int,
			default=100,
			help='How many images to train on at a time.')
	parser.add_argument(
			'--test_batch_size',
			type=int,
			default=-1,
			help="""\
			How many images to test on. This test set is only used once, to evaluate
			the final accuracy of the model after training completes.
			A value of -1 causes the entire test set to be used, which leads to more
			stable results across runs.""")
	parser.add_argument(
			'--validation_batch_size',
			type=int,
			default=100,
			help="""\
			How many images to use in an evaluation batch. This validation set is
			used much more often than the test set, and is an early indicator of how
			accurate the model is during training.
			A value of -1 causes the entire validation set to be used, which leads to
			more stable results across training iterations, but may be slower on large
			training sets.""")
	parser.add_argument(
			'--learning_rate',
			type=float,
			default=0.01,
			help='How large a learning rate to use when training.')
	parser.add_argument(
			'--how_many_training_steps',
			type=int,
			default=25000,
			help='How many training steps to run before ending.') #20000
	parser.add_argument(
			'--eval_step_interval',
			type=int,
			default=10,
			help='How often to evaluate the training results.')
	parser.add_argument(
			'--print_misclassified_test_images',
			default=False,
			help="""\
			Whether to print out a list of all misclassified test images.\
			""",
			action='store_true')

	FLAGS, unparsed= parser.parse_known_args()

	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)