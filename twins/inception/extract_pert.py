# encoding: utf-8
import os
import sys
import re

import scipy.io
import argparse

import tensorflow as tf
import numpy as np

FLAGS = None

def transform_pert(pert_dir, dest_dir, pert_name, extract_num):
	if pert_name == 'all':
		file_list = [x[2] for x in os.walk(pert_dir)][0]
	else:
		file_list = [pert_name]

	pattern = re.compile(r'[A-Za-z-0-9]+.mat$')
	for file in file_list:
		if pattern.match(file):
			file_name = file.split('.')[0]
			file_path = os.path.join(pert_dir, file)
			data = scipy.io.loadmat(file_path)

			dest_path = os.path.join(dest_dir, file_name+'_universal.npy')
			np.save(dest_path, data['r'])


def main(_):
	# start to transform .mat file to .npy file
	transform_pert(FLAGS.pert_dir, FLAGS.dest_dir, FLAGS.pert_name, FLAGS.extract_num)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--pert_dir', 
			type=str, 
			default='D:/Workspace/universal/precomputed', 
			help='Path to precomputed universal perturbations.')
	parser.add_argument(
			'--pert_name', 
			type=str, 
			default='all', 
			help='Name of .mat file that will extract perturbation from.')
	parser.add_argument(
			'--dest_dir', 
			type=str, 
			default='data/pert', 
			help='Destination path to transformed perturbations.')
	parser.add_argument(
			'--extract_num', 
			type=int, 
			default=3, 
			help='Number of perturbations wish to extract from the .mat file.')
	FLAGS, unparsed = parser.parse_known_args()

	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)