"""CIFAR-10评价

Accuracy:
	100k步训练（256个epoch）之后，cifar10_train.py能够达到83.0%的精确度
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import tf.app.flags.FLAGS

tf.app.flags.FLAGS.DEFINE_string('eval_dir', 'd:/Workspace/tensorflow/cifar10/cifar10_eval',
								"""Directory where to write event logs.""")
tf.app.flags.FLAGS.DEFINE_string('eval_data', 'test', 
								"""Either 'test' or 'train_eval'.""")
tf.app.flags.FLAGS.DEFINE_string('checkpoint_dir', 'd:/Workspace/tensorflow/cifar10/cifar10_train',
								"""Directory where to read model checkpoints.""")
tf.app.flags.FLAGS.DEFINE_integer('eval_interval_secs', 60 * 5, 
								"""How often to run the eval.""")
tf.app.flags.FLAGS.DEFINE_integer('num_examples', 10000, 
								"""Number of examples to run.""")
tf.app.flags.FLAGS.DEFINE_boolean('run_once', False, 
								"""Whether to run eval only once.""")

def eval_once(saver, summary_writer, top_k_op, summary_op):
	"""Run Eval once.

	Args:
		saver: Saver
		summary_writer: Summary writer
		top_k_op: Top k op
		summary_op: Summary op
	"""
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			#从检查点恢复
			saver.restore(sess, ckpt.model_checkpoint_path)
			#假设 model_checkpoint_path 形如: /my-favorite-path/cifar10_train/model.ckpt-0
			#从中提取global_step变量值
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

		#开启queue runner
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, 
												start=True))
			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0 #统计正确预测的数量
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1

			#计算precision @ 1
			precision = true_count / total_sample_count
			print('%s：precision @ 1 = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 1', simple_value=precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e: # pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

def evaluate():
	"""Eval CIFAR-10 for a number of steps."""
	with tf.Graph().as_default() as g:
		#获取图像和label样本
		eval_data = FLAGS.eval_data == 'test'
		images, labels = cifar10.inputs(eval_data=eval_data)

		#创建图，用推断模型计算logits predictions 
		logits = cifar10.inference(images)

		#计算预测值
		top_k_op = tf.nn.in_top_k(logits, labels, 1)
		#恢复学习到的变量的滑动平均值
		variable_averages = tf.train.ExponentialMovingAverage(
			cifar10.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		#基于Summaries集合创建summary op
		summary_op = tf.summary.merge_all()

		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

		while True:
			eval_once(saver, summary_writer, top_k_op, summary_op)
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)


def main(argv=None): # pylint: disable=unused-argument
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.eval_dir):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)


if __name__ == '__main__':
	tf.app.run()