# encoding: utf-8
"""
1. (optional)generate adversarial perturbation on a small subset of training set.
2. pick out examples which can be perturbated successfully.
3. cache bottleneck values of exampels that picked.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
# import shutil
import matplotlib.pyplot as plt
import sys, getopt
from urllib import request
import zipfile
from timeit import time

from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
from universal_pert import universal_perturbation


# =================================PICK RELATED=================================
def read_n_preprocess(image_path):
    """
    Helper funciton.
    """
    return preprocess_image_batch([image_path], img_size=(256, 256), crop_size=(224, 224))


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def save_pert(data_list, dest_path_list):
    for i in range(len(data_list)):
        value = ','.join(str(x) for x in data_list[i])
        with open(str(dest_path_list[i]), 'w') as pert_file:
            pert_file.write(value)


def pick_pert(sub_dirs, image_names, pert, f):
    """
    利用计算好的干扰筛选训练集,
    成功与失败的样本的文件名分别保存到.txt文件中:文件夹.txt(干扰成功) & 文件夹_fail.txt(干扰失败).
    txt文件数据格式：[文件夹名_图片名-干扰之后的label]。
    Args:
        sub_dirs: 保存有数据集根目录下所有文件夹完整路径的list
        image_names: 保存有每个dir下所有图片的名称
        pert: 预计算的干扰值
        f: 前馈函数
    """
    for i in range(len(sub_dirs)):        
        dir_name = os.path.basename(sub_dirs[i])
        succ_path = os.path.join('data/pick_pert', (str(dir_name) + '.txt'))
        fail_path = os.path.join('data/pick_pert', (str(dir_name) + '_fail.txt'))

        if os.path.isfile(succ_path):
            # os.remove(succ_path)
            continue
        elif os.path.isfile(fail_path):
            # os.remove(fail_path)
            continue

        perted_name = []
        fail_name = []
        cnt = 0

        prefix_path = sub_dirs[i]
        total_cnt = len(image_names[i])
        for j in range(total_cnt):
            img_path = os.path.join(prefix_path, image_names[i][j])
            image_original = read_n_preprocess(img_path) # return a array of dimension 4 
            # 计算裁剪后的干扰量，重新生成对抗样本
            clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:] + pert[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)
            image_perturbed = image_original + clipped_v[None, :, :, :]

            # 筛选干扰成功和失败的文件名
            # 文件名的格式为：【文件名】 + 【模型预测值】
            perted_pred = int(np.argmax(np.array(f(image_perturbed)).flatten()))
            if int(np.argmax(np.array(f(image_original)).flatten())) != perted_pred:
                perted_name.append(image_names[i][j] + '-' + str(perted_pred))、

                # 顺便计算bottleneck值，避免重复计算

                cnt += 1
            else:
                fail_name.append(image_names[i][j] + '-' + str(perted_pred))
        # 保存文件名和对应的label数据为.txt文件
        save_pert(data_list=[perted_name, fail_name], dest_path_list=[succ_path, fail_path])

        print('|++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('|++PICK PERT: Folder %s is processed successfully!!' % str(dir_name))
        print('|++Success Num: %d @ Fail Num: %d' % (cnt, (total_cnt - cnt)))


# =================================BOTTLENECK RELATED=================================
def create_image_lists(pert_dir):
    """Building a list of training images from the file system.
    Returns:
        result: filenames list of examples that can be perturbated successfully.
    """
    label_dir = 'data/ground_truth'
    ensure_dir_exists(label_dir)

    dirs = [x[0] for x in os.walk(pert_dir)]
    dirs = dirs[1:]
    Matrix = {}
    

    for d in dirs:
        basename = os.path.basename(d)
        Matrix[basename] = [x[2] for x in os.walk(d)][0]

    train_set_size = len(Matrix['train'])
    # 初始化用于计算bottleneck的训练样本
    result = [0 for x in range(train_set_size)]
    # ILSVRC2012数据集下有1000个类，每个类有1300个样本
    # label矩阵的每行的第一个元素用来记录该类样本label数
    # 只用于保存，不计入result列表中
    training_labels = np.zeros((train_set_size, 1301), dtype=np.int)

    # load n reorganize training set 
    for i, filename in enumerate(Matrix['train']):
        file_path = os.path.join(pert_dir, 'train', filename)
        training_images = []

        with open(file_path, 'r') as pert_file:
            pert_string = pert_file.read()
        pert_values = [str(x) for x in pert_string.split(',')]
        training_labels[i][0] = len(pert_values)

        for j, value in enumerate(pert_values):
            temp = value.split('-')
            training_images.append(str(temp[0]))
            training_labels[i][j+1] = int(temp[1])

        result[i] = {'dir': filename.split('.')[0], 
                        'training': training_images}
    np.save(os.path.join(label_dir, 'adv_train_labels.npy'), training_labels)

    # load n reorganize testing n validation set
    # test set size: 50,000, validation set size: 25,000
    for category in ['test', 'vali']:
        if category == 'test':
            labels = np.zeros((train_set_size, 51), dtype=np.int)
        elif category == 'vali':
            labels = np.zeros((train_set_size, 26), dtype=np.int)
        else:
            raise Exception('>>Create Image List: Invalid Category!!')

        for i, filename in enumerate(Matrix[category]):
            images = []
            file_path = os.path.join(pert_dir, category, filename)
            with open(file_path, 'r') as pert_file:
                pert_string = pert_file.read()
            pert_values = [str(x) for x in pert_string.split(',')]
            labels[i][0] = len(pert_values)

            for j, value in enumerate(pert_values):
                temp = value.split('-')
                images.append(str(temp[0]))
                labels[i][j+1] = int(temp[1])

            result[i][category] = images
        np.save(os.path.join(label_dir, 'adv_%s_labels.npy' % category), labels)

    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.
    Args:
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Int offset of the image we want.
        category: Name string of set to pull images from - training, testing, or
        validation.
    Returns:
        File system path string to an image that meets the requested parameters.
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
  
    base_name = category_list[index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)

    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, b_type=0):
    """Returns a path to a bottleneck file for a label at the given index.
    Args:
        category: Name string of set to pull images from - training, testing, or
    validation.
        b_type: Type String of bottleneck values - norm(0) or adv(!=0).
    Returns:
        File system path string to an bottleneck that requested.
        e.g. 'bottleneck/n0157841/train/norm_1.txt'
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
  
    sub_dir = label_lists['dir']
    bottleneck_type = 'norm'
    if b_type != 0:
        bottleneck_type = 'adv'

    return os.path.join(bottleneck_dir, sub_dir, category, '%s_%d.txt' % (bottleneck_type, index))


def run_bottleneck_on_image(sess, image_data, input_tensor, bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Returns:
        Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(bottleneck_tensor, feed_dict={input_tensor : image_data})
    return np.squeeze(bottleneck_values)


def create_bottleneck_file(bottleneck_path, image_dir, image_lists, label_name, index,
                             category, bottleneck_type, sess, input_tensor, bottleneck_tensor):
    if os.path.exists(bottleneck_path):
        os.remove(bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('>>Cant create bottleneck: Image does not exist %s', image_path)

    image_data = read_n_preprocess(image_path)
    if bottleneck_type != 0:
        # 添加对抗干扰
        clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:] + pert[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)
        image_data = image_data + clipped_v[None, :, :, :]

    bottleneck_values = run_bottleneck_on_image(sess, image_data, input_tensor, bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index,
                                        image_dir, category, bottleneck_dir,
                                        input_tensor, bottleneck_tensor, bottleneck_type):
    """Retrieves or calculates bottleneck values for an image.
    Args:
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Integer offset of the image we want. This will be modulo-ed by the available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string  of the subfolders containing the training images.
        category: Name string of which  set to pull images from(training, testing, or validation)
    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, bottleneck_type)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_dir, image_lists, label_name, index, category, bottleneck_type, sess, input_tensor, bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except:
        print("Invalid float found, recreating bottleneck")
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_dir, image_lists, label_name, index, category, bottleneck_type, sess, input_tensor, bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def cache_bottlenecks(sess, image_lists, bottleneck_dir, input_tensor, bottleneck_tensor):
    """Ensures all the training, testing, and validation bottlenecks are cached.
    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
     """
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)

    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            how_many_bottlenecks = 0
            for index, _ in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index,
                                        image_dir, category, bottleneck_dir,
                                        input_tensor, bottleneck_tensor, bottleneck_type=0)
                get_or_create_bottleneck(sess, image_lists, label_name, index,
                                        image_dir, category, bottleneck_dir,
                                        input_tensor, bottleneck_tensor, bottleneck_type=1)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 1000 == 0:
                    print('>>Folder %s, %d %s bottleneck files created.' % 
                            (label_name, how_many_bottlenecks, category))


# =================================MAIN CODE=================================
if __name__ == '__main__':
    # Parse arguments
    argv = sys.argv[1:]

    # Default values
    PATH_TRAIN_IMAGENET = 'D:/Scholarship/dataset/ILSVRC2012'
    # PATH_TRAIN_IMAGENET = 'D:/workspace/img'
    PATH_TEST_IMAGE = 'data/test_img.png'
    PATH_PERT = 'data/pick_pert'

    try:
        opts, args = getopt.getopt(argv, "i:t:p:", ["test_image=", "training_path=", "pert_path="])
    except getopt.GetoptError:
        print ('python ' + sys.argv[0] + ' -i <test image> -t <imagenet training path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-t':
            PATH_TRAIN_IMAGENET = arg
        if opt == '-i':
            PATH_TEST_IMAGE = arg
        if opt == '-p':
            PATH_PERT = arg

    persisted_sess = tf.Session()
    # inception_model_path = os.path.join('D:/Workspace/tensorflow/twins/inception/inception_pretrain', 
    #                                         'classify_image_graph_def.pb')
    inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')

    if os.path.isfile(inception_model_path) == 0:
        print("Downloading Inception model...")
        request.urlretrieve ("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip", os.path.join('data', 'inception5h.zip'))
        # request.urlretrieve ("http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz")
        # Unzipping the file
        zip_ref = zipfile.ZipFile(os.path.join('inception_pretrain', 'inception-2015-12-05.tgz'), 'r')
        zip_ref.extract('classify_image_graph_def.pb', 'inception_pretrain')
        zip_ref.close()
        tarfile.open(filepath, 'r:gz').extractall('inception_pretrain')

    model = os.path.join(inception_model_path)

    # Load the Inception model 
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    persisted_input = persisted_sess.graph.get_tensor_by_name("input:0") # 299x299x3
    # print(persisted_input)
    persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")
    # print(persisted_output)
    persisted_bottleneck = persisted_sess.graph.get_tensor_by_name("avgpool0/reshape:0")

    print(">> Computing feedforward function...")
    def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={
                                    persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})

    file_perturbation = os.path.join('data', 'universal.npy')

    if not os.path.exists(file_perturbation):

        # TODO: Optimize this construction part!
        print(">> Compiling the gradient tensorflow functions. This might take some time...")
        scalar_out = [tf.slice(persisted_output, [0, i], [1, 1]) for i in range(0, 1000)]
        #因为gradient计算的是所有输出w.r.t每一个输入特征的导数之和，不能直接得到雅可比矩阵形式的结果，所以必须每次取出一个输出值，分别计算关于输入的导数，最后组装起来
        dydx = [tf.gradients(scalar_out[i], [persisted_input])[0] for i in range(0, 1000)]

        print(">> Computing gradient function...")
        def grad_fs(image_inp, inds): return [persisted_sess.run(dydx[i], feed_dict={
                                    persisted_input: image_inp}) for i in inds]

        # Load/Create data
        datafile = os.path.join('data', 'imagenet_data.npy')
        if os.path.isfile(datafile) == 0:
            print(">> Creating pre-processed imagenet data...")
            #预处理一个batch的图片并返回
            X, dirs, sub_dirs = create_imagenet_npy(PATH_TRAIN_IMAGENET, 1000)
            print(X.shape)

            # print(">> Saving the pre-processed imagenet data")
            # if not os.path.exists('data'):
            #     os.makedirs('data')

            # Save the pre-processed images
            # Caution: This can take take a lot of space. Comment this part to discard saving.
            # np.save(os.path.join('data', 'imagenet_data.npy'), X)

        else:
            print(">> Pre-processed imagenet data detected")
            X = np.load(datafile)

        #核心部分！！计算通用干扰
        # Running universal perturbation
        v = universal_perturbation(X, f, grad_fs, delta=0.2)

        # Saving the universal perturbation
        np.save(os.path.join(file_perturbation), v)

    else:
        print(">> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation)
        v = np.load(file_perturbation)

    if not os.path.exists(PATH_PERT):
        dirs = [x[0] for x in os.walk(PATH_TRAIN_IMAGENET)]
        dirs = dirs[1:]

        dirs = sorted(dirs)
        Matrix = [0 for x in range(1000)]
        it = 0

        for d in dirs:
            for _, _, filename in os.walk(d):
                Matrix[it] = filename
            it += 1

        # pick out and save examples that can be perturbed successfully 
        pick_pert(dirs, Matrix, v, f)

    sess.close()

    #===========================Test the perturbation on the image========================
    # print(">> Testing the universal perturbation on an image")

    # labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')

    # image_original = preprocess_image_batch([PATH_TEST_IMAGE], img_size=(299, 299), color_mode="rgb")
    # label_original = np.argmax(f(image_original), axis=1).flatten()

    # str_label_original = labels[np.int(label_original)-1].split(',')[0] #?????

    # # Clip the perturbation to make sure images fit in uint8
    # #undo_image_avg()：将规则化的图片复原，3个通道都加上平均值
    # clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)

    # print(v.shape) # [1,224,224,3]
    # print(clipped_v.shape) #[224,224,3]

    # image_perturbed = image_original + clipped_v[None, :, :, :]
    # label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
    # str_label_perturbed = labels[np.int(label_perturbed)-1].split(',')[0]


    # #=================================Show original and perturbed image======================================
    # plt.figure()
    # #subplot(nrows, ncols, plot_number)
    # plt.subplot(1, 2, 1)
    # plt.imshow(undo_image_avg(image_original[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
    # plt.title(str_label_original)

    # plt.subplot(1, 2, 2)
    # plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
    # plt.title(str_label_perturbed)

    # plt.show()