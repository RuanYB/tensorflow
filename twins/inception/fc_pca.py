#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from bro_classifier import create_graph
from bottleneck_util import run_bottleneck_on_image

BOTTLENECK_TENSOR_NAME = 'avgpool0/reshape:0'
BOTTLENECK_TENSOR_SIZE = 1024
ORIN_FC_TENSOR_NAME = 'softmax2_pre_activation:0'
SHADOW_FC_TENSOR_NAME = 'final_training_ops/Wx_plus_b/add:0'
FC_TENSOR_SIZE = 1008


def compute_fc_value(graph_dir, graph_name, bltnk_values, return_elements):
	"""returns a numpy array consists of model's corresponding full-connect layer outputs
	"""
	with tf.Graph().as_default():
		data_size = bltnk_values.shape[0]
		fc_values = np.zeros((data_size, FC_TENSOR_SIZE), dtype=np.float32)
		graph, (btlnk_tensor, fc_tensor) = create_graph(
												graph_dir=graph_dir, 
												graph_name=graph_name, 
												return_elements=return_elements)
		print('++ fc_tensor shape: ', fc_tensor.shape)
		num_batches = int(np.ceil(data_size / FLAGS.batch_size))

		with tf.Session() as sess:
			for i in range(num_batches):
				start = i * FLAGS.batch_size
				end = min((i+1)*FLAGS.batch_size, data_size)
				logits = run_bottleneck_on_image(sess, bltnk_values[start:end], 
												btlnk_tensor, fc_tensor)
				fc_values[start : end] = logits
				if start % 1000 == 0:
					print('++ %d bottlenecks has been processed' % start)
	return fc_values 


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--path_bltnk',
		type=str,
		default='data/bottleneck/vali',
		help='path to pre-computed bottleneck values.')
	parser.add_argument(
		'--model_dir',
		type=str,
		default='data/graph',
		help='Directory where to store and retrieve the pre-computed model graph.')
	parser.add_argument(
		'--graph_name',
		type=str,
		default='tensorflow_inception_graph.pb',
		help='name of graph about to load.')
	parser.add_argument(
		'--path_eigenvec',
		type=str,
		default='data/eigen_vectors.npy',
		help='Path to cache file which stored pre-computed eigenvectors.')
	parser.add_argument(
		'--path_eigenval',
		type=str,
		default='data/eigen_values.npy',
		help='Path to cache file which stored pre-computed eigenvalues.')
	parser.add_argument(
		'--batch_size',
		type=int,
		default=100,
		help='number of bottlenecks to be feed.')
	parser.add_argument(
		'--num_eigen',
		nargs='+',
		type=int,
		help='order number of eigenvectors wants to analyse.')
	parser.add_argument(
		'--ver',
		type=bool,
		default=False,
		help='whether to plot variance explained ratio.')
	FLAGS, _ = parser.parse_known_args()
	if(len(FLAGS.num_eigen) != 2) raise ValueError('!!Error: must offer only 2 eigenvector order number.')
	print('++ order number of eigenvectors about to project: ', FLAGS.num_eigen)
	return_elements = [BOTTLENECK_TENSOR_NAME, ORIN_FC_TENSOR_NAME] if FLAGS.graph_name == 'tensorflow_inception_graph.pb' else [BOTTLENECK_TENSOR_NAME, SHADOW_FC_TENSOR_NAME]


	if (not os.path.exists(FLAGS.path_eigenvec)) or (not os.path.exists(FLAGS.path_eigenval)): 
		# load validation bottleneck values: sum up 50000
		bltnk_filenames = [x[2] for x in os.walk(FLAGS.path_bltnk)][0]
		bltnk_values = np.zeros((len(bltnk_filenames), BOTTLENECK_TENSOR_SIZE), dtype=np.float32)
		for i, filename in enumerate(bltnk_filenames):
			filename = os.path.join(FLAGS.path_bltnk, filename)
			with open(filename, 'r') as bltnk_file:
				bltnk_string = bltnk_file.read()
				bltnk_value = [float(x) for x in bltnk_string.split(',')]
			bltnk_values[i] = bltnk_value
			if i+1==50000:
				print('>>Validation set processing complete!')


		# load original inception model
		print('>>Load specific graph...')
		fc_values = compute_fc_value(FLAGS.path_bltnk, FLAGS.graph_name,
										bltnk_values, return_elements)
		print('++ Fc_values shape: ', fc_values.shape)

		# remove the mean
		sc = StandardScaler(copy=False, with_std=False)
		fc_values = sc.fit_transform(fc_values)

		# compute covariance matrix
		cov_mat = np.cov(fc_values)
		print('++ Covariance matrix shape: ', cov_mat.shape)

		# compute eigenvector n eigenvalue of covariance matrix
		# column vecs[:, i] is the eigenvector corresponding to the eigenvalue val[i]
		eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
		print('++ Eigen vector shape: ', eigen_vecs.shape)	

		# compute ratio of every eigenvalue in ensemble(descending order)
		eigen_mask = np.argsort(-eigen_vals, axis=1)
		eigen_vals =  eigen_vals[eigen_mask]
		eigen_vecs = eigen_vecs[:, eigen_mask]

		# store the pre-computed eigenvectors and eigenvalues
		np.save(FLAGS.path_eigenvec, eigen_vecs)
		np.save(FLAGS.path_eigenval, eigen_vals)
	else:
		eigen_vals =  np.load(FLAGS.path_eigenval)
		eigen_vecs = np.load(FLAGS.path_eigenvec)
	print('++ Eigen_val shape: ', eigen_vals.shape)
	print('++Eigen_vecs shape: ', eigen_vecs.shape)
	print('FOR DEBUG -- eigen mask:', eigen_mask)
	print('FOR DEBUG -- eigen values', eigen_vals)
	
	if FLAGS.ver:
		# plot eigenvalue distribution
		total = sum(eigen_vals)
		var_exp = [(i / total) for i in eigen_vals]
		num_bar = int(len(eigen_vals) / 20)
		# 方差解释率 (variance explained ratios) 
		plt.bar(range(num_bar), var_exp[:num_bar], width=1.0, bottom=0.0, label='individual explained variance')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.legend(loc='best')
		plt.show()

	# feature transformation
	# extract project matrix: shape 2x2
	w = eigen_vecs[:, FLAGS.num_eigen]
	print('++ Project matrix shape: ', w.shape)
	# transform validation set data
	fc_values_pca = fc_values.dot(w)
	print('++ fc_values_pca shape: ', fc_values_pca.shape)

	# plot scatter plot of original n adversarial examples projected to specific dimensions
	batch_fc_vals = int(fc_values_pca.shape[0] / 2)
	colors = ['b', 'r']
	markers = ['o','x']
	dataset_mask = [1, 0]
	for l, c , m in zip(dataset_mask, colors, markers):
		plt.scatter(fc_values_pca[l*batch_fc_vals : (l+1)*batch_fc_vals, 0], 
					fc_values_pca[l*batch_fc_vals : (l+1)*batch_fc_vals, 1], 
					c=c, 
					label='normal examples' if l == 1 else 'adversarial examples',
					marker=m)
		plt.xlabel('Eigenvector No.%d' % FLAGS.num_eigen[0])
		plt.ylabel('Eigenvector No.%d' % FLAGS.num_eigen[1])
		plt.legend(loc='lower left')
		plt.show()