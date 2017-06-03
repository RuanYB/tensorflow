# encoding: utf-8
"""
Concatenate module mnist and mnist_train
"""
import tensorflow as tf
import numpy as np

from datetime import datetime
import time
import os

import mnist_input
from utils import weight_variable, bias_variable, conv2d, max_pool, softmax_loss, train_accuracy, train

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 20000, 
							"""Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10, 
							"""How often to log results to the console.""")
tf.app.flags.DEFINE_integer('batch_size', 50, 
							"""Size of a batch of examples.""")
tf.app.flags.DEFINE_string('data_dir', 'D:/Workspace/tensorflow/twins/mnist_for_twins/data', 
							"""Directory where to store or retrieve data.""")
tf.app.flags.DEFINE_string('train_dir', 'D:/Workspace/tensorflow/twins/mnist_for_twins/data/mnist_train', 
							"""Directory where to write event logs and checkpoints.""")

MIN_AFTER_DEQUEUE = 1000
TFRECORDS_TYPE = "data"

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
	file_path = os.path.join(TFRECORDS_TYPE, '%s.tfrecords' % data_type)
	if not os.path.exists(file_path):
		mnist_input.generate_tfrecords(data_type)
	if 'test' == data_type:
		shuffle = False
	image, label = mnist_input.parse_tfrecords(tfrecord_type=data_type)
	return mnist_input.generate_image_and_label_batch(
						image, label, MIN_AFTER_DEQUEUE, batch_size=50, shuffle=shuffle)


def main(argv=None):
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)

	global_step = tf.contrib.framework.get_or_create_global_step()
	save_path = os.path.join(FLAGS.train_dir, 'model_ckpt')

	#获取(image, label)batch pair
	image_batch, label_batch = inputs(data_type='train')

	#损失函数sparse_softmax_cross_entropy_with_logits要求rank_of_labels = rank_of_images - 1
	#对label_batch作扁平化处理
	label_batch = tf.reshape(label_batch, [50])

	#扩展image维度，从[batch, row, col]转换为[batch, row, col, depth=1]
	expand_image_batch = tf.expand_dims(image_batch, -1)

	input_placeholder = tf.placeholder_with_default(expand_image_batch, shape=[None, 28, 28, 1], name='input')


	# 构建模型
	# 第一个卷积层
	with tf.variable_scope('conv1') as scope:
		kernal = weight_variable('weights', shape=[5, 5, 1 ,32])
		biases = bias_variable('biases', shape=[32])
		pre_activation = tf.nn.bias_add(conv2d(input_placeholder, kernal), biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	# 第一个池化层
	pool1 = max_pool(conv1)

	# 第二个卷积层
	with tf.variable_scope('conv2') as scope:
		kernal = weight_variable('weights', shape=[5, 5, 32, 64])
		biases = bias_variable('biases', shape=[64])
		pre_activation = tf.nn.bias_add(conv2d(pool1, kernal), biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	# 第二个池化层
	# 7*7*64
	pool2 = max_pool(conv2)

	# 全连接层
	with tf.variable_scope('fc1') as scope:
		weight_fc1 = weight_variable('weights', shape=[7*7*64, 1024])
		biases = bias_variable('biases', shape=[1024])
		pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
		fc1 = tf.nn.relu((tf.matmul(pool2_flat, weight_fc1) + biases), name=scope.name)
		print('Tensor fc1/relu: ', fc1.name)

	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	fc1_drop = tf.nn.dropout(fc1, keep_prob, name='fc1_drop')
	print('>>Tensor dropout: ', fc1_drop.name)

	# 输出层
	with tf.variable_scope('softmax_linear') as scope:
		weight_fc2 = weight_variable('weight', shape=[1024, 10])
		biases = bias_variable('biases', shape=[10])

		softmax_output = tf.add(tf.matmul(fc1_drop, weight_fc2), biases, name=scope.name)
		print('>>Tensor softmax_linear/softmax_output: ', softmax_output.name)


	loss = softmax_loss(logits=softmax_output, labels=label_batch)
	print('>>Tensor loss: ', loss.name)

	accuracy = train_accuracy(softmax_output, label_batch)
	print('>>Tensor accuracy: ', accuracy.name)

	train_op = train(loss, global_step)
	print('>>Tensor train_op: ', train_op.name)

	#初始化所有参数
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		coord =tf.train.Coordinator()
		try:
			threads = []

			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord, 
											daemon=True, start=True))

			saver = tf.train.Saver()
			step = 1

			while step <= 20000 and not coord.should_stop():
				if step % 100 == 0:
					# 每隔100步打印一次accuracy
					runtime_accuracy = sess.run(accuracy, feed_dict={keep_prob: 1.0})
					print(">>step %d, training accuracy %g" % (step, runtime_accuracy))

					# 每隔1000步保存一次模型
					if step % 1000 == 0:
						saver.save(sess, save_path, global_step=step)

				# 训练模型
				sess.run(train_op, feed_dict={keep_prob: 0.5})
				# 步数更新
				step += 1
		except Exception as e:
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)
	# ========================HOOK & SESSION RUN CODE=============================
	# class _LoggerHook(tf.train.SessionRunHook):
	# 	"""
	# 	记录损失和运行时间日志信息
	# 	"""
	# 	def begin(self):
	# 		self._step = -1
	# 		self._start_time = time.time()

	# 	def before_run(self, run_context):
	# 		self._step += 1
	# 		# self._start_time = time.time()
	# 		#请求目标tensor的值，在after_run方法中获取
	# 		return tf.train.SessionRunArgs([loss, accuracy])

	# 	def after_run(self, run_context, run_values):
	# 		if self._step % FLAGS.log_frequency == 0:
	# 			_current_time = time.time()
	# 			duration = _current_time - self._start_time

	# 			self._start_time = _current_time

	# 			#提取before_run中请求的损失和精确度值
	# 			loss_value, accuracy_value = run_values.results
	# 			#样本数/秒，秒/batch_size数样本
	# 			examples_per_sec = FLAGS.batch_size * FLAGS.log_frequency / duration
	# 			sec_per_batch = float(duration / FLAGS.log_frequency)

	# 			#console打印训练状态数据
	# 			#时间：步数，损失，精确度（每秒样本数，每batch样本处理时间） 
	# 			format_str = ('%s: step %d, loss=%.2f, accuracy=%.2f(%.1f examples/sec, %.3f sec/batch)')
	# 			print(format_str % (datetime.now(), self._step, loss_value, accuracy_value, examples_per_sec, sec_per_batch))


	# 辣鸡！！！！！！
	# 最大训练步数20000，每10步打印一次输出
	# MonitoredTrainingSession默认情况下600s保存一次检查点，每100步保存一次summary
	# with tf.train.MonitoredTrainingSession(
	# 	checkpoint_dir=FLAGS.train_dir,
	# 	hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
	# 			tf.train.NanTensorHook(loss),
	# 			_LoggerHook()],
	# 	save_checkpoint_secs=60,
	# 	config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as mon_sess:
	# 	mon_sess.run(init)
	# 	while not mon_sess.should_stop():
	# 		mon_sess.run(train_op, feed_dict={keep_prob: 0.5})


# ===============LAUNCH CODE================
if __name__ == '__main__':
	tf.app.run()