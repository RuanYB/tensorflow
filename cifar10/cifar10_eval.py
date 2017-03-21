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
tf.app.flags.FLAGS.DEFINE_string('checkpoint_dir', 'd:/Workspace/tensorflow/cifar10/cifar10_train')
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
