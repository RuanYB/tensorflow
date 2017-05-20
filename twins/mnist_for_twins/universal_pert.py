# encoding: utf-8
import numpy as np

from deepfool import deepfool

def proj_lp(v, xi, p):
	"""
	Project on the lp ball centered at 0 and of radius xi
	SUPPORTS only p = 2 and p = Inf for now
	"""
	if p == 2:
	    v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
	    # v = v / np.linalg.norm(v.flatten(1)) * xi
	elif p == np.inf:
	    v = np.sign(v) * np.minimum(abs(v), xi) # 无穷范数即是求元素绝对值的最大值,此处检查每个元素是否大于半径xi
	else:
	     raise ValueError('Values of p different from 2 and Inf are currently not supported...')

	return v


def universal_perturbation(dataset, ff, grads, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
	"""
	:param dataset: Images of size MxHxWxC (M: number of images)

	:param ff: feedforward function (input: images, output: values of activation BEFORE softmax).

	:param grads: gradient functions with respect to input (as many gradients as classes).

	:param delta: controls the desired fooling rate (default = 80% fooling rate)

	:param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

	:param xi: controls the l_p magnitude of the perturbation (default = 10)

	:param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

	:param num_classes: num_classes (limits the number of classes to test against, by default = 10)

	:param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

	:param max_iter_df: maximum number of iterations for deepfool (default = 10)

	:return: the universal perturbation.
	"""
	v = 0
	fooling_rate = 0.0
	num_images = np.shape(dataset)[0]

	itr = 0

	while fooling_rate < 1-delta and itr < max_iter_uni:
		# 打乱数据集
		np.random.shuffle(dataset)

		print('Starting pass number: %d......' % itr)

		# 遍历数据集并依序计算干扰增量
		for k in range(num_images):
			image = dataset[k:(k+1), :, :, :]

			# 当前干扰量无法改变当前图片分类时，计算干扰增量
			if(int(np.argmax(np.array(ff(image)).flatten())) == int(np.argmax(np.array(ff(image+v)).flatten()))):
				print('>>当前iter:# %d, 样本编号: %d' % (itr, k))

				# 计算对抗干扰
				# func(Deelfool) return: 
				# minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
				dr, df_iter, _, _ = deepfool(image + v, ff, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

				# 确保deepfool算法收敛，收敛则更新通用干扰增量
				if df_iter < max_iter_df-1:
					v += dr
					# Project on l_p ball, restrict lp magnitude of v
					v = proj_lp(v, xi, p)

			itr += 1

			# 使用当前干扰量对整个数据子集施加干扰
			dataset_perturbed = dataset + v

			# 初始化矩阵：存储样本被干扰前后的label值
			pred_orig_labels = np.zeros(num_images)
			pred_pert_labels = np.zeros(num_images)

			eval_size = 100
			num_evals = int(np.ceil(float(num_images) / float(eval_size)))

			for i in range(num_evals):
				