# encoding: utf-8

import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import mnist

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 20000, 
							"""Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10, 
							"""How often to log results to the console.""")
# tf.app.flags.DEFINE_integer('checkpoint_frequency', 1000,
# 							"""How often to save checkpoint to the train_dir.""")
tf.app.flags.DEFINE_integer('batch_size', 50, 
							"""Size of a batch of examples.""")
tf.app.flags.DEFINE_string('train_dir', 'D:/Workspace/tensorflow/twins/mnist_for_twins/data/mnist_train', 
							"""Directory where to write event logs and checkpoints.""")

#DUMPED CODE
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
	# with tf.Graph().as_default():
	# 	global_step = tf.contrib.framework.get_or_create_global_step()
	global_step = tf.contrib.framework.get_or_create_global_step()

	#初始化所有参数
	init = tf.global_variables_initializer()

	#获取(image, label)batch pair
	image_batch, label_batch = mnist.inputs('train')

	#损失函数sparse_softmax_cross_entropy_with_logits要求rank_of_labels = rank_of_images - 1
	#对label_batch作扁平化处理
	label_batch = tf.reshape(label_batch, [50])

	#扩展image维度，从[batch, row, col]转换为[batch, row, col, depth=1]
	expand_image_batch = tf.expand_dims(image_batch, -1)

	#损失函数使用sparse_softmax_cross_entropy_with_logits()，自动完成one_hot编码转化
	#将label数据由标量转换为one_hot编码形式
	# labels_one_hot = dense_to_one_hot(label_batch, 10)

	#创建mnist模型，并计算每个batch样本的logits
	logits = mnist.inference(expand_image_batch, dropout=0.5)

	loss = mnist.loss(logits=logits, labels=label_batch)

	accuracy = mnist.train_accuracy(logits, label_batch)

	train_op = mnist.train(loss, global_step)


	class _LoggerHook(tf.train.SessionRunHook):
		"""
		记录损失和运行时间日志信息
		"""
		def begin(self):
			self._step = -1
			self._start_time = time.time()

		def before_run(self, run_context):
			self._step += 1
			# self._start_time = time.time()
			#请求目标tensor的值，在after_run方法中获取
			return tf.train.SessionRunArgs([loss, accuracy])

		def after_run(self, run_context, run_values):
			if self._step % FLAGS.log_frequency == 0:
				_current_time = time.time()
				duration = _current_time - self._start_time

				self._start_time = _current_time

				#提取before_run中请求的损失和精确度值
				loss_value, accuracy_value = run_values.results
				#样本数/秒，秒/batch_size数样本
				examples_per_sec = FLAGS.batch_size * FLAGS.log_frequency / duration
				sec_per_batch = float(duration / FLAGS.log_frequency)

				#console打印训练状态数据
				#时间：步数，损失，精确度（每秒样本数，每batch样本处理时间） 
				format_str = ('%s: step %d, loss=%.2f, accuracy=%.2f(%.1f examples/sec, %.3f sec/batch)')
				print(format_str % (datetime.now(), self._step, loss_value, accuracy_value, examples_per_sec, sec_per_batch))


	#最大训练步数20000，每10步打印一次输出
	#MonitoredTrainingSession默认情况下600s保存一次检查点，每100步保存一次summary
	with tf.train.MonitoredTrainingSession(
		checkpoint_dir=FLAGS.train_dir,
		hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
				tf.train.NanTensorHook(loss),
				_LoggerHook()],
		save_checkpoint_secs=60,
		config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as mon_sess:
		mon_sess.run(init)
		while not mon_sess.should_stop():
			# mon_sess.run(init)
			mon_sess.run(train_op)


def main(argv=None):
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)

	train()

if __name__ == '__main__':
	tf.app.run()