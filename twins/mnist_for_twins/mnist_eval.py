# encoding: utf-8

"""
MNIST评价
测试数据集样本数：10000
"""
import tensorflow as tf
import numpy as np

import math
from datetime import datetime

import mnist

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'data/mnist_eval',
									""""Directory where to write event logs""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'data/mnist_train',
									"""Directory where to read model checkpoints""")
tf.app.flags.DEFINE_integer('num_examples', 10000, 
									"""Number of examples to test""")
tf.app.flags.DEFINE_integer('batch_size', 50,
									"""Number of examples to test in a batch""")

def evaluate():
	image_batch, label_batch = mnist.inputs(data_type='test')

	image_batch = tf.expand_dims(image_batch, -1)
	label_batch = tf.reshape(label_batch, [50])

	#default value of dropout:1.0
	logits = mnist.inference(image_batch) 

	#计算预测值
	top_k_op = tf.nn.in_top_k(logits, label_batch, 1)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		#从检查点恢复模型
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print(ckpt.model_checkpoint_path)
			#提取global_step变量值
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No Checkpoint File Found!')
			return 

		#开启queue runner
		coord = tf.train.Coordinator()
		try:
			threads = []
			#为数据读取队列创建线程
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord, 
												daemon=True, start=True))

			#防止样本总数无法被batch_size整除，batch()默认不接受小于设定大小的batch输出情况
			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			total_example_cnt = num_iter * FLAGS.batch_size
			true_cnt = 0 #统计正确预测的数量
			step = 0

			while step < num_iter and not coord.should_stop():
				predictions = sess.run(top_k_op)
				true_cnt += np.sum(predictions)
				step += 1

			#计算测试精确度:precision @ 1
			precision =  true_cnt / total_example_cnt
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

			#参考的code中还有用于tensorboard的summary存储操作
		except Exception as e:
			coord.request_stop(e)

		coord.request_stop()
		#当一个线程调用了request_stop()方法后，其余线程有grace period的时间来停止(default 2min)
		#超时后由coordinator.join()报运行时异常
		coord.join(threads, stop_grace_period_secs=10)


def main(argv=None): 
	evaluate()

if __name__ == '__main__':
	tf.app.run()