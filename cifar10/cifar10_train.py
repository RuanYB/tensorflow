"""使用单GPU训练CIFAR-10网络

Accuracy：
	经cifar_eval.py评估，100k步训练后（256个epoch），cifar10_train.py的精确度可以达到 ~86%.
Speed：
	batch_size 128

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'd:/Workspace/tensorflow/cifar10/cifar10_train',
							"""Directory where to write event logs"""
							"""and checkpoint""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, 
							"""Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, 
							"""Whether to log device placement""")
tf.app.flags.DEFINE_integer('log_frequency', 10, 
							"""How often to log results to the console.""")

def train():
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()

	#为CIFAR-10获取（图像，label）样本对
	images, labels = cifar10.distorted_inputs()

	#创建图Graph用于从推断模型计算logits predictions
	logits = cifar10.inference(images)

	#计算损失
	loss = cifar10.loss(logits, labels)

	#创建图：用一个batch的样本来训练模型
	#更新模型参数
	train_op = cifar10.train(loss, global_step)

	class _LoggerHook(tf.train.SessionRunHook):
		"""Logs loss and runtime"""

		def begin(self):
			self._step = -1
			self._start_time = time.time()

		def before_run(self, run_context):
			self._step += 1
			return tf.train.SessionRunArgs(loss) #请求loss值

		def after_run(self, run_context, run_values):
			if self._step % FLAGS.log_frequency == 0:
				current_time = time.time()
				duration = current_time - _start_time
				self._start_time = current_time

				loss_value = run_values.results
				examples_per_sec = FLAGS.batch_size * log_frequency / duration
				sec_per_batch = float(duration / log_frequency)

				format_str = ('%s: step %d, loss = %.2f(%.1f examples/sec, %.3f sec/batch)')
				print(format_str % (datetime.now(), self._step, loss_value, 
						examples_per_sec, sec_per_batch))

	with tf.train.MonitoredTrainingSession(
		checkpoint_dir=FLAGS.train_dir,
		hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
				tf.train.NanTensorHook(loss),
				_LoggerHook()],
		config=tf.ConfigProto(
			log_device_placement = FLAGS.log_device_placement)) as mon_sess:
		while not mon_sess.should_stop():
			mon_sess.run(train_op)

def main(argv=None): #pylint: disable=unused-argument
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	train()


if __name__ == '__main__':
	tf.app.run()