# encoding: utf-8
"""
对抗样本集构造
"""
import argparse

import tensorflow as tf

from mnist import inputs
from universal_pert import universal_perturbation
from mnist_input import generate_fix_data_batch
import mnist_train

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', 'data/mnist_train', 
							"""Directory where to import the model graph""")


def generate_adv_exmp(pool_size, dataset_size):
	# 从检查点提取模型路径
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	model_path = ckpt.model_checkpoint_path
	#元数据meta路径
	model_meta_path = '.'.join([model_path, 'meta'])

	# 加载模型
	new_saver = tf.train.import_meta_graph(model_meta_path, clear_devices=True)

	with tf.Session() as sess:
		# 恢复模型参数
		new_saver.restore(sess, model_path)

		# 从Graph中提取模型的输入和输出tensor
		# input[50, 28, 28, 1], softmax[50, 10]
		input_op = tf.get_collection('input_op')[0]
		softmax_linear_op = tf.get_collection('softmax_linear_op')[0]

		# gradient计算的是所有输出w.r.t每一个输入特征的导数之和，不能直接得到雅可比矩阵形式的结果，
		# 所以必须每次取出一个输出值，分别计算关于输入的导数，最后组装起来
		scalar_out = [tf.slice(softmax_linear_op, [0, i], [1, 1]) for i in range(10)]
		dydx = [tf.gradients(scalar_out[i], [input_op])[0] for i in range(10)] # why extract [0]?

		def ff(image_inp):
			"""
			前馈导数计算函数
			"""
			return sess.run(softmax_linear_op, feed_dict={
									input_op: np.reshape(image_inp, (-1, 28, 28, 1))})

		def grads(image_inp, inds):
			"""
			梯度计算函数
			"""
			return [sess.run(dydx[i], feed_dict={input_op: image_inp}) for i in inds]

		# 获取样本池（默认500），并随机抽取size_dataset的样本，提供给对抗干扰生成函数
		image_pool, label_pool = generate_fix_data_batch(sess=sess, pool_size=pool_size)
		index_for_pool = np.arange(poolsize)
		np.random.shuffle(index_for_pool)
		images = image_pool[index_for_pool[:dataset_size]]
		labels = label_pool[index_for_pool[:dataset_size]]

		# 根据样本子集构造universal adversarial perturbation
		v = universal_perturbation(images, ff, grads, delta=0.2)

		return v


# def _test_pert(v):
	"""
	测试计算而得干扰量的有效性
	"""



# =================MAIN FUNC & LAUNCH CODE==================
def main(argv=None):
	#检查点存储目录检查
	if not tf.gfile.Exists(FLAGS.model_dir):
		mnist_train.train()

	dataset_size = 100
	pool_size = 500

	# Parse console arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--dataset', type=int, 
						help='Number of examples to construct adversarial perturbation')
	parser.add_argument('-p', '--poolsize', type=int,
						help='Size of pool that generated for random sampling')
	args = parser.parse_args()

	if args.dataset:
		dataset_size = args.dataset
		print('>>ARG UPDT: Num of exmps to construct adv perb changed to %d...' % dataset_size)
	elif args.poolsize:
		pool_size = args.poolsize
		print('>>ARG UPDT: Size of pool for random sampling changed to %d...' % pool_size)		

	# 使用指定size的样本集生成对抗干扰
	pert_v = generate_adv_exmp(pool_size, dataset_size)

	# _test_pert(v)


if __name__ == "__main__":
	tf.app.run()