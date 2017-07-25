import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
# import shutil
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
from urllib import request
import zipfile

from timeit import time

from universal_pert import universal_perturbation

# JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
# RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

def read_n_preprocess(image_path):
    """
    Helper funciton.
    """
    return preprocess_image_batch([image_path], img_size=(256, 256), crop_size=(224, 224))


def save_pert(data_list, dest_path_list):
    for i in range(len(data_list)):
        value = ','.join(str(x) for x in data_list[i])
        with open(str(dest_path_list[i]), 'w') as pert_file:
            pert_file.write(value)


def pick_pert(sub_dirs, image_names, pert, f):
    """
    利用计算好的干扰筛选训练集,
    成功与失败的样本的文件名分别保存到.txt文件中.
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
                perted_name.append(image_names[i][j] + '-' + str(perted_pred))

                cnt += 1
            else:
                fail_name.append(image_names[i][j] + '-' + str(perted_pred))
        # 保存文件名和对应的label数据为.txt文件
        save_pert(data_list=[perted_name, fail_name], dest_path_list=[succ_path, fail_path])

        print('|++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('|++PICK PERT: Folder %s is processed successfully!!' % str(dir_name))
        print('|++Success Num: %d @ Fail Num: %d' % (cnt, (total_cnt - cnt)))


if __name__ == '__main__':
    # Parse arguments
    argv = sys.argv[1:]

    # Default values
    PATH_TRAIN_IMAGENET = 'D:/Scholarship/dataset/ILSVRC2012'
    # PATH_TRAIN_IMAGENET = 'D:/workspace/img'
    PATH_TEST_IMAGE = 'data/test_img.png'
    PATH_PERT = 'data/pick_pert'

    try:
        opts, args = getopt.getopt(argv, "i:t:", ["test_image=", "training_path="])
    except getopt.GetoptError:
        print ('python ' + sys.argv[0] + ' -i <test image> -t <imagenet training path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-t':
            PATH_TRAIN_IMAGENET = arg
        if opt == '-i':
            PATH_TEST_IMAGE = arg

    persisted_sess = tf.Session()
    # inception_model_path = os.path.join('D:/Workspace/tensorflow/twins/inception/inception_pretrain', 
    #                                         'classify_image_graph_def.pb')
    inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')

    if os.path.isfile(inception_model_path) == 0:
        print("Downloading Inception model...")
        # request.urlretrieve ("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip", os.path.join('data', 'inception5h.zip'))
        request.urlretrieve ("http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz")
        # Unzipping the file(这里应该是untgz，待修改)
        # zip_ref = zipfile.ZipFile(os.path.join('inception_pretrain', 'inception-2015-12-05.tgz'), 'r')
        # zip_ref.extract('classify_image_graph_def.pb', 'inception_pretrain')
        # zip_ref.close()
        tarfile.open(filepath, 'r:gz').extractall('inception_pretrain')

    model = os.path.join(inception_model_path)

    # Load the Inception model 
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')


    persisted_input = persisted_sess.graph.get_tensor_by_name("input:0") # 299x299x3
    print(persisted_input)
    persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")
    print(persisted_output)

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


    #================================Test the perturbation on the image========================================
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