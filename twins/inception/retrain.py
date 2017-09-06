# encoding: utf-8
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.python import debug as tf_debug

from bottleneck_util import get_bottleneck_path, load_npy

FLAGS = None

DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'avgpool0/reshape:0'
BOTTLENECK_TENSOR_SIZE = 1024
MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_DEPTH = 3
PRE_ACTIVATION_TENSOR = 'softmax2_pre_activation:0'
INPUT_TENSOR = 'input:0'


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# ============================ BOTTLENECK RELATED ============================
def create_bottleneck_lists(pert_dir, ground_truth_dir):
  """Builds a list of bottleneck files from the file system.
  Analyzes the file names in the pick directory, and retrieve label number of each class.
  Returns a data structure describing the list of labels and corresponding quantity.
  Args:
    pert_dir: String path to a folder containing pickpert files of subfolders.
    ground_truth_dir: String path to npy files holding label infos.
  Returns:
    A dictionary containing an entry for each label subfolder, with quantity of 
    training, testing, and validation sets within each label.
  """
  path_train_pert = os.path.join(pert_dir, 'train')
  filenames = [x[2] for x in os.walk(path_train_pert)][0]
  bottleneck_list = []

  # load label name and store as a directory
  for filename in filenames:
    label_name = filename.split('.')[0]
    bottleneck_list.append({'dir' : label_name})

  # load quantity of train, test or vali set of each label
  bltlk_len = len(bottleneck_list)
  for category in ['train', 'test', 'vali']:
    labels = load_npy(ground_truth_dir, category)
    for index in range(bltlk_len):
        bottleneck_list[index][category] = labels[index][0]

  return bottleneck_list


def get_bottleneck(bottleneck_lists, label_index, bottleneck_index, bottleneck_dir, 
                      category, bottleneck_type):
  bottleneck_path = get_bottleneck_path(bottleneck_lists, label_index, bottleneck_index, 
                                          bottleneck_dir, category, bottleneck_type)
  if not os.path.exists(bottleneck_path):
    print('>>Bottleneck file doesnt exist: %s!' % os.path.basename(bottleneck_path))
    return None
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

  return bottleneck_values, bottleneck_path


def get_random_cached_bottlenecks(bottleneck_lists, how_many, category, category_labels,
                                    bottleneck_dir):
  """Retrieves bottleneck values for cached images.
  It picks a random set of bottlenecks from the specified category.
  Args:
    bottleneck_lists: Dictionary of training images' bottleneck quatity for each label.
    how_many: If positive, a random sample of this size will be chosen.
              If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - train, test or vali.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
  # initialize variables
  class_count = len(bottleneck_lists)
  bottlenecks = []
  ground_truths = []
  filenames = []

  if how_many >= 0:
    cnt = 0
    # Retrieve a random sample of bottlenecks.
    for _ in range(int(how_many/2)):
      label_index = random.randrange(class_count) # randrange返回指定范围内的一个随机数
      # label_name = bottleneck_lists[label_index]['dir']
      bottleneck_index = random.randrange(bottleneck_lists[label_index][category])
      # bottleneck_type = False if random.randrange(2)==0 else True # python中三元运算符的形式(0:norm, 1:adv)
      # bottleneck, bottleneck_path = get_bottleneck(bottleneck_lists, label_index, bottleneck_index, 
      #                                         bottleneck_dir, category, bottleneck_type)
      bottleneck_1, bottleneck_path_1 = get_bottleneck(bottleneck_lists, label_index, bottleneck_index, 
                                        bottleneck_dir, category, True)
      bottleneck_2, bottleneck_path_2 = get_bottleneck(bottleneck_lists, label_index, bottleneck_index, 
                                        bottleneck_dir, category, False)
      if bottleneck_1 and bottleneck_2:
        bottlenecks.append(bottleneck_1)
        bottlenecks.append(bottleneck_2)
        # retrieve label saved in disk
        ground_truth = category_labels[label_index][bottleneck_index+1]
        ground_truths.append(ground_truth)
        ground_truths.append(ground_truth)
        filenames.append(bottleneck_path_1)
        filenames.append(bottleneck_path_2)
        cnt += 2

    # print('>>Get random %d %s Bottlenecks and %d labels' % (len(bottlenecks), category, len(ground_truths)))
    if cnt != how_many:
      print('>>Get random %d %s Bottlenecks: Failure count add up to %d!' % (how_many, category, (how_many - cnt)))
  else:
    # Retrieve all bottlenecks.
    cnt = 0
    for label_index in range(class_count):
      bottleneck_list = bottleneck_lists[label_index]
      for bottleneck_index in range(bottleneck_list[category]):
        adv_bottleneck, adv_bottleneck_path = get_bottleneck(bottleneck_lists, label_index, bottleneck_index, 
                                              bottleneck_dir, category, True)
        norm_bottleneck, norm_bottleneck_path = get_bottleneck(bottleneck_lists, label_index, bottleneck_index, 
                                              bottleneck_dir, category, False)
        cnt += 2
        if norm_bottleneck:
          bottlenecks.append(norm_bottleneck)
          ground_truth = category_labels[label_index][bottleneck_index+1]
          ground_truths.append(ground_truth)
          filenames.append(norm_bottleneck_path)
        if adv_bottleneck:
          bottlenecks.append(adv_bottleneck)
          ground_truth = category_labels[label_index][bottleneck_index+1]
          ground_truths.append(ground_truth)
          filenames.append(adv_bottleneck_path)

    # print('>>Get random %d %s Bottlenecks and %d labels' % (len(bottlenecks), len(ground_truths)))
    if len(ground_truths) != cnt:
      print('>>Total Num: %d, Actual Num: %d' % (cnt, len(ground_truths)))
      print('>>Get All %s Bottlenecks: Failure count add up to %d!' % (category, (len(ground_truths) - cnt)))

  return bottlenecks, ground_truths, filenames


# ============================ NETWORK RELATED ============================
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained network, and various tensors we'll be manipulating.
  """
  print('>>Creating graph...')
  with tf.Session() as sess:
    model_filename = os.path.join(
        FLAGS.model_dir, 'tensorflow_inception_graph.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, input_tensor, output_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
                                BOTTLENECK_TENSOR_NAME, INPUT_TENSOR,
                                PRE_ACTIVATION_TENSOR]))
  return sess.graph, bottleneck_tensor, input_tensor, output_tensor


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
  """Adds a new softmax and fully-connected layer for training.
  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.
  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  print('>>Adding final training ops...')
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInput')

    ground_truth_input = tf.placeholder(tf.int64, [None], name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=ground_truth_input, 
        logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy_mean)

  return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.
  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.
  Returns:
    Tuple of (evaluation step, prediction).
  """
  print('>>Adding evaluation steps...')
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(prediction, tf.cast(ground_truth_tensor, tf.int64))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def save_model(sess, graph, step, save_labels=False):
  # Write out the trained graph and labels with the weights stored as constants.
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  output_graph_name = os.path.join(FLAGS.model_dir, 're_retrain_%d_graph.pb' % step)
  with gfile.FastGFile(output_graph_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())

  if save_labels:
    output_labels = [x['dir'] for x in bottleneck_lists]
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(output_labels) + '\n')


def main(_):
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # Set up the pre-trained graph.
  graph, bottleneck_tensor, input_tensor, output_tensor = create_inception_graph()

  # Look at the folder structure, and create lists of all the images.
  bottleneck_lists = create_bottleneck_lists(FLAGS.pert_dir, FLAGS.ground_truth_dir)
  class_count = len(bottleneck_lists)
  print('>>%d valid class of bottlenecks found at %s' % (class_count, FLAGS.pert_dir))

  sess = tf.Session()

  # type order ‘run -f has_inf_or_nan’ in commandline to launch the hook
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

  # Add the new layer that we'll be training.
  train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor = add_final_training_ops(
                                            1008,
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

  # Create the operations we need to evaluate the accuracy of our new layer.
  evaluation_step, prediction = add_evaluation_step(final_tensor, 
                                                      ground_truth_input)

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  # Set up all our weights to their initial default values.
  init = tf.global_variables_initializer()
  sess.run(init)

  # retrieve category labels from the cache stored on disk
  train_labels = load_npy(FLAGS.ground_truth_dir, 'train', True)
  vali_labels = load_npy(FLAGS.ground_truth_dir, 'vali', True)

  # Run the training for as many cycles as requested on the command line.
  for i in range(FLAGS.how_many_training_steps):
    # Get a batch of input bottleneck values from the cache stored on disk.
    train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
        bottleneck_lists, FLAGS.train_batch_size, 'train', train_labels,
        FLAGS.bottleneck_path)

    # Feed the bottlenecks and ground truth into the graph, and run a training
    # step. Capture training summaries for TensorBoard with the`merged`op.
    train_summary, _ = sess.run([merged, train_step],
             feed_dict={bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth})
    train_writer.add_summary(train_summary, i)

    if (i % FLAGS.save_model_interval == 0) and (i + 1 >= 3500):
      save_model(sess, sess.graph, i, save_labels=False)

    # Every so often, print out how well the graph is training.
    is_last_step = (i + 1 == FLAGS.how_many_training_steps)
    if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
      train_accuracy, cross_entropy_value = sess.run(
          [evaluation_step, cross_entropy],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                      train_accuracy * 100))
      print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                 cross_entropy_value))
      vali_bottlenecks, vali_ground_truth, _ = get_random_cached_bottlenecks(
              bottleneck_lists, FLAGS.validation_batch_size, 'vali', vali_labels,
              FLAGS.bottleneck_path)
      # Run a validation step anddata/bottleneckng summaries for TensorBoard
      # with the `merged` op.
      validation_summary, validation_accuracy = sess.run(
          [merged, evaluation_step],
          feed_dict={bottleneck_input: vali_bottlenecks,
                     ground_truth_input: vali_ground_truth})
      validation_writer.add_summary(validation_summary, i)
      print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
            (datetime.now(), i, validation_accuracy * 100,
             len(vali_bottlenecks)))

  # We've completed all our training, so run a final test evaluation on
  # some new images we haven't used before.
  test_bottlenecks, test_ground_truth, test_filenames = (
      get_random_cached_bottlenecks(bottleneck_lists, FLAGS.test_batch_size,
                                    'test', FLAGS.bottleneck_path))
  test_accuracy, predictions = sess.run(
      [evaluation_step, prediction],
      feed_dict={bottleneck_input: test_bottlenecks,
                 ground_truth_input: test_ground_truth})
  print('Final test accuracy = %.1f%% (N=%d)' % (
      test_accuracy * 100, len(test_bottlenecks)))

  # if FLAGS.print_misclassified_test_images:
  #   print('=== MISCLASSIFIED TEST IMAGES ===')
  #   for i, test_filename in enumerate(test_filenames):
  #     if predictions[i] != test_ground_truth[i]:
  #       print('%70s  %s' % (test_filename,
  #                           list(image_lists.keys())[predictions[i]]))

  # Write out the trained graph and labels with the weights stored as constants.
  # output_graph_def = graph_util.convert_variables_to_constants(
  #     sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  # with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
  #   f.write(output_graph_def.SerializeToString())
  # output_labels = [x['dir'] for x in bottleneck_lists]
  # with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
  #   f.write('\n'.join(output_labels) + '\n')
  save_model(sess, sess.graph, FLAGS.how_many_training_steps, save_labels=True)  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--pert_dir',
      type=str,
      default='data/pick_pert',
      help='Path to pick pert files.'
  )
  parser.add_argument(
      '--ground_truth_dir',
      type=str,
      default='data/ground_truth',
      help='Path to ground truth files.'
  )
  parser.add_argument(
      '--bottleneck_path',
      type=str,
      default='data/bottleneck',
      help='Path to ground truth files.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='data/graph/retrain_output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='data/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='data/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=30000,
      help='How many training steps to run before ending.'
  ) # default: 4000
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--save_model_interval',
      type=int,
      default=500,
      help='How often to save the graph as pb file.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  # parser.add_argument(
  #     '--print_misclassified_test_images',
  #     default=False,
  #     help="""\
  #     Whether to print out a list of all misclassified test images.\
  #     """,
  #     action='store_true'
  # )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='data/graph',
      help="""\
      Path to model graph file.
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='data/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)