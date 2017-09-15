# encoding: utf-8
"""
Bottleneck related helper functions.
"""
import sys

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
import matplotlib.pyplot as plt

def load_npy(ground_truth_dir, category, adv_or_not=True):
  """
  Args:
  ground_truth_dir: String path to npy files holding label infos
  category: Type string of bottleneck file(train, test or vali)
  adv_or_not: Type boolean of whether retrieving adversarial labels or not
  Returns:
  Numpy array of labels of specific category
  """
  if adv_or_not:
    bottleneck_type = 'adv'
  else:
    bottleneck_type = 'orin'
  file_path = os.path.join(ground_truth_dir, '%s_%s_labels.npy' % (bottleneck_type, category))
  try:
    labels = np.load(file_path)
  except IOError:
    print('>>Error: file %s doesn\'t exists!' % file_path)
    sys.exit(2)
  # print('>>Retrieved %s labels from the cache stored in disk successfully!' % category)

  return labels

def get_image_path(image_lists, label_index, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.
    Args:
        image_lists: List of training images for each label.
        label_index: Label Number int we want to get an image for.
        index: Int offset of the image we want.
        category: Name string of set to pull images from - training, testing, or
        validation.
    Returns:
        File system path string to an image that meets the requested parameters.
    """
    if label_index >= len(image_lists):
        tf.logging.fatal('Label does not exist %s.', label_index)
    label_lists = image_lists[label_index]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_index, category)
  
    file_name = category_list[index]
    if category == 'train':
        sub_dir = label_lists['dir']
        full_path = os.path.join(image_dir, category, sub_dir, file_name)
    else:
        full_path = os.path.join(image_dir, category, file_name)

    return full_path


def get_bottleneck_path(image_lists, label_index, index, bottleneck_dir, category, b_type=True):
    """Returns a path to a bottleneck file for a label at the given index.
    Args:
        category: Name string of set to pull images from - train, test, or vali.
        b_type: Type String of bottleneck values - norm(0) or adv(!=0).
    Returns:
        File system path string to an bottleneck that requested.
        e.g. 'bottleneck/train/norm_n0157841_1.txt'
    """
    # label_name = image_lists[label_index]['dir']
    # if label_index >= len(image_lists):
    #     tf.logging.fatal('Label index does not exist: ', label_index)
    # if category not in label_lists:
    #     tf.logging.fatal('Category does not exist: ', category)
    # category_list = label_lists[category]
    # if not category_list:
    #     tf.logging.fatal('Label %s has no images in the category %s.',
    #                      label_name, category)

    label_name = image_lists[label_index]['dir']
    bottleneck_type = 'norm'
    if b_type:
        bottleneck_type = 'adv'

    return os.path.join(bottleneck_dir, category, '%s_%s_%d.txt' % (bottleneck_type, label_name, index))


def run_bottleneck_on_image(sess, input_data, input_tensor, bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Returns:
        Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(bottleneck_tensor, feed_dict={input_tensor : input_data})
    return np.squeeze(bottleneck_values)


def create_bottleneck_file(bottleneck_path, image_dir, image_lists, label_index, index,
                             category, bottleneck_type, sess, input_tensor, bottleneck_tensor, pert):
    if os.path.exists(bottleneck_path):
        os.remove(bottleneck_path)
    image_path = get_image_path(image_lists, label_index, index, image_dir, category)
    # print('>>Create Bottleneck File——Extract Image From Path: %s' % image_path)
    if not gfile.Exists(image_path):
        tf.logging.fatal('>>Cant create bottleneck: Image does not exist %s', image_path)

    image_data = read_n_preprocess(image_path)
    if bottleneck_type != 0:
        # 添加对抗干扰
        clipped_v = np.clip(undo_image_avg(image_data[0,:,:,:] + pert[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_data[0,:,:,:]), 0, 255)
        image_data = image_data + clipped_v[None, :, :, :]

    bottleneck_values = run_bottleneck_on_image(sess, image_data, input_tensor, bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_index, index,
                                        image_dir, category, bottleneck_dir,
                                        input_tensor, bottleneck_tensor, pert, bottleneck_type):
    """Retrieves or calculates bottleneck values for an image.
    Args:
        image_lists: List of training images for each label.
        label_index: Label Number int we want to get an image from.
        index: Integer offset of the image we want. This will be modulo-ed by the available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string  of the subfolders containing the training images.
        category: Name string of which  set to pull images from(train, test, or vali)
        bottleneck_type: Type string of bottleneck, 0 for normal, not 0 for adversarial.
    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_index]
    sub_dir = label_lists['dir']
    # check if folder train or test or vali exists
    sub_dir_path = os.path.join(bottleneck_dir, category)
    ensure_dir_exists(sub_dir_path)

    bottleneck_path = get_bottleneck_path(image_lists, label_index, index, bottleneck_dir, category, bottleneck_type)
    if not os.path.exists(bottleneck_path):
        print('>>Bottleneck file doesnt exist, starting creating: ', bottleneck_path)
        create_bottleneck_file(bottleneck_path, image_dir, image_lists, label_index, index, category, 
                                bottleneck_type, sess, input_tensor, bottleneck_tensor, pert)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except:
        print("Invalid float found, recreating bottleneck")
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_dir, image_lists, label_index, index, category, 
                                bottleneck_type, sess, input_tensor, bottleneck_tensor, pert)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, input_tensor, bottleneck_tensor, pert):
    """Ensures all the training, testing, and validation bottlenecks are cached.
    Args:
        sess: The current active TensorFlow Session.
        image_lists: List of training images for each label, each element is a directory.
     """
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)

    for label_index, label_lists in enumerate(image_lists):
        print('>>Starting Cache Folder: %s' % image_lists[label_index]['dir'])
        for category in ['train', 'test', 'vali']:
            category_list = label_lists[category]
            how_many_bottlenecks = 0
            for index, _ in enumerate(category_list):
                # generate corresponding bottleneck file of original n adversarial example
                get_or_create_bottleneck(sess, image_lists, label_index, index,
                                        image_dir, category, bottleneck_dir,
                                        input_tensor, bottleneck_tensor, pert, bottleneck_type=True)
                get_or_create_bottleneck(sess, image_lists, label_index, index,
                                        image_dir, category, bottleneck_dir,
                                        input_tensor, bottleneck_tensor, pert, bottleneck_type=False)

                how_many_bottlenecks += 1
            print('>>Folder %s: %d %s bottleneck files has been created!!' % 
                        (label_lists['dir'], how_many_bottlenecks, category))


def jsd_analyze(jsd_list):
    # former half part  of jsd_list are adversarial js divergence
    # latter half part are normal ones
    half_size = int(jsd_list.shape[0]/2)
    x_axis = np.array(range(half_size), dtype=np.int)
    print('++ jsd_analyze -- x_axis length:', len(x_axis))
    colors = ['b', 'r']
    marker = ['o', 'x']
    list_mask = [1, 0]
    for l, c, m in zip(list_mask, colors, marker):
        plt.scatter(x_axis, 
                    jsd_list[l*half_size : (l+1)*half_size],
                    c=c,
                    label='normal' if l == 1 else 'adversarial',
                    marker=m)
    plt.xlabel('Dimension')
    plt.ylabel('JS Divergence')
    plt.legend(loc='best')
    plt.show()