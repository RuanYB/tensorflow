# encoding: utf-8
"""
对抗样本集构造
"""
import tensorflow as tf

from mnist import inputs
from universal_pert import universal_perturbation
import mnist_train

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', 'data/mnist_train', 
							"""Directory where to import the model graph""")


def generate_adv_exmp():
	image_batch, label_batch = inputs(data_type='train')

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
		# print(sess.run(scalar_out, feed_dict={input_op: image_batch}))
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

		# 根据样本子集构造universal adversarial perturbation
		v = universal_perturbation(dataset, ff, grads, delta=0.2)


# =================MAIN FUNC & LAUNCH CODE==================
def main(argv=None):
	#检查点存储目录检查
	if not tf.gfile.Exists(FLAGS.model_dir):
		mnist_train.train()
	# 生成对抗样本集
	generate_adv_exmp()


if __name__ == "__main__":
	tf.app.run()