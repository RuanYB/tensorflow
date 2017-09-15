# encoding: utf-8
"""融合original net和shadow net的结果
1. js divergence效果不佳，不如top法
2. cosine待测试...
"""

import os
import argparse
import numpy as np
from bottleneck_util import jsd_analyze

p = argparse.ArgumentParser()
# p.add_argument('-p', '--pp', type=int, nargs='+', help='')
p.add_argument('-t', '--temperature', type=float, default=0.01, help='阈值.')
p.add_argument('-n', '--file_name', type=str, default='', help='jsd file cache stored in disk.')
p.add_argument('-e', '--eval_type', type=str, default='jsd', help='type of evaluation(jsd or cos).')
p.add_argument('--plot', type=bool, default=False, help='whether to plot the scatter diagram.')

FLAGS, _ = p.parse_known_args()

path_jsd_data = os.path.join('data/pca', FLAGS.file_name)
data = np.load(path_jsd_data)
print('>> js divergence data shape: ', data.shape)

half_size = int(data.shape[0]/2)
adv_part = data[:half_size]
norm_part = data[half_size:]
if FLAGS.eval_type == 'jsd':
	adv_cnt = sum(adv_part > FLAGS.temperature)
	norm_cnt = sum(norm_part <= FLAGS.temperature)
elif FLAGS.eval_type == 'cos':
	adv_cnt = sum(adv_part > FLAGS.temperature)
	norm_cnt = sum(norm_part <= FLAGS.temperature)	
print('adv count: %d, percentage: %.2f%%' % (adv_cnt, 100*adv_cnt/half_size))
print('norm count: %d, percentage: %.2f%%' % (norm_cnt, 100*norm_cnt/half_size))
if FLAGS.plot:
	jsd_analyze(data)