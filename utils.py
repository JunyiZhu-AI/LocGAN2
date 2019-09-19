from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
import numpy as np
import random
import os
import multiprocessing
from tqdm import tqdm
import math
from PIL import Image
import pandas as pd
import shutil
from functools import partial, reduce


def image_input(source_dir, with_gridmap=True, image_pattern='png', gridmap_pattern='png'):
    '''
    reading all the images and grid maps in the source dir and pairing.
    :return: path to paired image and grid map saved in tf dataset
    '''
    # get all file path
    birdview_train_fn = np.array(glob.glob(r'' + source_dir + '/train/images/*.' + image_pattern))
    birdview_test_fn = np.array(glob.glob(r'' + source_dir + '/test/images/*.' + image_pattern))

    if with_gridmap:
        gridmap_train_fn = np.array(glob.glob(r'' + source_dir + '/train/gridmaps/*.' + gridmap_pattern))
        gridmap_test_fn = np.array(glob.glob(r'' + source_dir + '/test/gridmaps/*.' + gridmap_pattern))

        # sort file paths according to name, in order to align image and gridmap
        birdview_train_fn = sorted(birdview_train_fn, key=file_index)
        gridmap_train_fn = sorted(gridmap_train_fn, key=file_index)
        birdview_test_fn = sorted(birdview_test_fn, key=file_index)
        gridmap_test_fn = sorted(gridmap_test_fn, key=file_index)
        return tf.data.Dataset.from_tensor_slices({
            'image': birdview_train_fn,
            'gridmap': gridmap_train_fn
        }), tf.data.Dataset.from_tensor_slices({
            'image': birdview_test_fn,
            'gridmap': gridmap_test_fn
        })
    return tf.data.Dataset.from_tensor_slices({'image': birdview_train_fn}), \
           tf.data.Dataset.from_tensor_slices({'image': birdview_test_fn})


def file_index(filename):
    return int(os.path.splitext(os.path.basename(filename))[0])


def task_net_input(source_dir, pos_weight, load_list, image_pattern='png', gridmap_pattern='png',
                   only_test=False):
    if not load_list:
        image_test_list, gridmap_test_list, name_test_list, gt_test_list, loc_test_list, \
        x_test_list, y_test_list, a_test_list, image_train_list, gridmap_train_list, \
        name_train_list, gt_train_list, loc_train_list, x_train_list, y_train_list, a_train_list = \
            generator_dataset_list(source_dir=source_dir, image_pattern=image_pattern,
                                   gridmap_pattern=gridmap_pattern, pos_weight=pos_weight,
                                   only_test=only_test, save_list=False)

    else:
        train_df = pd.read_pickle(source_dir + '/list_train')
        test_df = pd.read_pickle(source_dir + '/list_test')
        image_test_list, gridmap_test_list, name_test_list, gt_test_list, loc_test_list, \
        x_test_list, y_test_list, a_test_list = np.array(test_df['image'].tolist()), \
                                                np.array(test_df['gridmap'].tolist()), \
                                                np.array(test_df['name'].tolist()), \
                                                np.array(test_df['gt'].tolist()), \
                                                np.array(test_df['loc'].tolist()), \
                                                np.array(test_df['x'].tolist()), \
                                                np.array(test_df['y'].tolist()), \
                                                np.array(test_df['angle'].tolist())

        image_train_list, gridmap_train_list, name_train_list, gt_train_list, loc_train_list, \
        x_train_list, y_train_list, a_train_list = np.array(train_df['image'].tolist()), \
                                                   np.array(train_df['gridmap'].tolist()), \
                                                   np.array(train_df['name'].tolist()), \
                                                   np.array(train_df['gt'].tolist()), \
                                                   np.array(train_df['loc'].tolist()), \
                                                   np.array(train_df['x'].tolist()), \
                                                   np.array(train_df['y'].tolist()), \
                                                   np.array(train_df['angle'].tolist())

    if not only_test:
        print('### loaded instaces: {} ###'.format(image_test_list.shape[0] + image_train_list.shape[0]))
        return tf.data.Dataset.from_tensor_slices({
            'image': image_train_list,
            'gridmap': gridmap_train_list,
            'gt': gt_train_list,
            'loc': loc_train_list,
            'name': name_train_list,
            'x': x_train_list,
            'y': y_train_list,
            'a': a_train_list
        }), tf.data.Dataset.from_tensor_slices({
            'image': image_test_list,
            'gridmap': gridmap_test_list,
            'gt': gt_test_list,
            'loc': loc_test_list,
            'name': name_test_list,
            'x': x_test_list,
            'y': y_test_list,
            'a': a_test_list,
        })
    else:
        print('###loaded instaces: {} ###'.format(image_test_list.shape[0]))
        return None, tf.data.Dataset.from_tensor_slices({
            'image': image_test_list,
            'gridmap': gridmap_test_list,
            'gt': gt_test_list,
            'loc': loc_test_list,
            'name': name_test_list,
            'x': x_test_list,
            'y': y_test_list,
            'a': a_test_list,
        })


def data_augmentation(image_dict):
    '''
    Data augmentation includes rotation, cropping, flipping, random brightness, random contrast
    '''
    print('### data augumentation on ###')
    # add random brightness and random contra to birdview image
    image_dict['image'] = tf.image.random_brightness(image_dict['image'], max_delta=20.0 / 255.0)
    image_dict['image'] = tf.image.random_contrast(image_dict['image'], lower=0.7, upper=1.3)

    ####################parameters for crop & rotation#########################
    # rotation
    with tf.variable_scope('rotate_image'):
        rot_rad = tf.random.uniform(shape=[], minval=-math.pi / 2, maxval=math.pi / 2, dtype=tf.float32)
        # scale according to rotation
        r_scale = math.sqrt(2) / (2 * tf.math.sin(math.pi / 4 + tf.math.abs(rot_rad)))
        ul = 0.5 - r_scale / 2
        br = ul + r_scale
    # crop
    with tf.variable_scope('crop_image'):
        c_scale = tf.random.uniform(shape=[], minval=0.7, maxval=1.0, dtype=tf.float32)
        top = tf.random.uniform(shape=[], minval=0, maxval=1 - c_scale, dtype=tf.float32)
        left = tf.random.uniform(shape=[], minval=0, maxval=1 - c_scale, dtype=tf.float32)
        bottom = top + c_scale
        right = left + c_scale

    ####################parameters for crop & rotation#########################

    # rotate images
    def rotate_images(image):
        with tf.name_scope('rotat_image'):
            image = tf.contrib.image.rotate(images=image, angles=rot_rad, interpolation='BILINEAR')
            image = tf.image.crop_and_resize(tf.expand_dims(image, axis=0),
                                             boxes=[[ul, ul, br, br]], box_ind=[0],
                                             crop_size=image.shape.as_list()[0:2])
            image = tf.squeeze(input=image, axis=0)
        return image

    # crop images
    def crop_images(image):
        with tf.name_scope('crop_image'):
            image = tf.image.crop_and_resize(tf.expand_dims(image, axis=0),
                                             boxes=[[top, left, bottom, right]], box_ind=[0],
                                             crop_size=image.shape.as_list()[0:2])
            image = tf.squeeze(input=image, axis=0)
        return image

    # rotate or crop images
    rand_crop_rotate = tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32), 0.5)
    for k, v in image_dict.items():
        if k not in ['gridmap', 'image']:
            image_dict[k] = v
            continue
        image_dict[k] = tf.where(condition=rand_crop_rotate, x=rotate_images(v), y=crop_images(v))

    # flip left right
    rand_left_right = tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32), 0.5)
    for k, v in image_dict.items():
        if k not in ['gridmap', 'image']:
            image_dict[k] = v
            continue
        image_dict[k] = tf.where(condition=rand_left_right, x=tf.image.flip_left_right(v), y=v)

    # flip up down
    rand_up_down = tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32), 0.5)
    for k, v in image_dict.items():
        if k not in ['gridmap', 'image']:
            image_dict[k] = v
            continue
        image_dict[k] = tf.where(condition=rand_up_down, x=tf.image.flip_up_down(v), y=v)

    return image_dict


def load_image(img_pattern='png', img_width=500, img_height=500, grayscale=True, augumentation=True):
    '''
    second step on input pipline. Loading images and preprocessing
    :return: function to load the image which will be used in dataset map function.
    '''
    if grayscale:
        channel = 1
    else:
        channel = 3

    # wrap the load image function
    def _load_image(path_dict, pattern='', width=500, height=500):
        image_dict = {}
        for k, v in path_dict.items():
            if k not in ['gridmap', 'image']:
                image_dict[k] = v
                continue
            # read in image
            image_string = tf.read_file(v)
            if pattern == 'png':
                image = tf.image.decode_png(image_string, channels=channel)
            elif pattern == 'jpeg':
                image = tf.image.decode_jpeg(image_string, channels=channel)
            else:
                print('Unknow image format!')
                exit(1)
            # convert values to [0, 1] range
            image = tf.image.convert_image_dtype(image, tf.float32)
            # resize the image
            image = tf.image.resize_images(image, [height, width])
            image_dict[k] = image
        if augumentation:
            image_dict = data_augmentation(image_dict)
        return image_dict

    return partial(_load_image, pattern=img_pattern, width=img_width, height=img_height)


def list_pb(x, image_pattern, pos_weight):
    birdview_fn, gridmap_fn, shift_subdir = x
    image_list = np.array([])
    x_list = np.array([])
    y_list = np.array([])
    a_list = np.array([])
    # hold label lists'shape
    loc_list = np.array([[0, 0]])
    gt_list = np.array([[0, 0]])

    subdir_files = np.array(glob.glob(r'' + shift_subdir + '/*'))
    image_files = np.array([x for x in subdir_files if image_pattern in x])
    image_files = sorted(image_files, key=file_index)
    gridmap_list = np.array([gridmap_fn] * (len(image_files) + 1 * pos_weight))
    name_list = np.array([os.path.splitext(os.path.basename(gridmap_fn))[0]] * (len(image_files) + 1 * pos_weight))

    for i in range(pos_weight - 1):
        image_list = np.append(image_list, [birdview_fn], axis=0)
    info_dir = ''
    for x in subdir_files:
        if image_pattern not in x:
            info_dir = x
    infos = np.loadtxt(info_dir)
    for image, info in zip(image_files, infos):
        assert int(os.path.splitext(os.path.basename(image))[0]) == info[0], 'pairing shifted image failed'
        image_list = np.append(image_list, image)
        x_list = np.append(x_list, info[5])
        y_list = np.append(y_list, info[6])
        a_list = np.append(a_list, info[8])
        gt_list = np.append(gt_list, np.expand_dims(info[1:3], axis=0), axis=0)
        loc_list = np.append(loc_list, np.expand_dims(info[3:5], axis=0), axis=0)

    # delete first element of allocator of label list
    gt_list = gt_list[1:, :]
    loc_list = loc_list[1:, :]

    return image_list, gridmap_list, name_list, gt_list, loc_list, x_list, y_list, a_list


def generator_dataset_list(source_dir, image_pattern, gridmap_pattern, pos_weight, only_test, save_list):
    def make_list(birdview_fns, gridmap_fns, shifted_subdirs, screen):
        gridmap_list = np.array([])
        image_list = np.array([])
        x_list = np.array([])
        y_list = np.array([])
        a_list = np.array([])
        name_list = np.array([])
        # hold label lists'shape
        loc_list = np.array([[0, 0]])
        gt_list = np.array([[0, 0]])

        _list_pb = partial(list_pb, image_pattern=image_pattern, pos_weight=pos_weight)
        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=int(cores / 8)) as p:
            with tqdm(total=len(birdview_fns), desc="generate dataset list", disable=not screen) as pbar:
                for im, g, n, gt, loc, x, y, a in p.imap(_list_pb, zip(birdview_fns, gridmap_fns, shifted_subdirs)):
                    image_list = np.append(image_list, im, axis=0)
                    gridmap_list = np.append(gridmap_list, g, axis=0)
                    name_list = np.append(name_list, n, axis=0)
                    gt_list = np.append(gt_list, gt, axis=0)
                    loc_list = np.append(loc_list, loc, axis=0)
                    x_list = np.append(x_list, x, axis=0)
                    y_list = np.append(y_list, y, axis=0)
                    a_list = np.append(a_list, a, axis=0)
                    pbar.update()
            # delete first element of allocator of label list
            gt_list = gt_list[1:, :]
            loc_list = loc_list[1:, :]

        return image_list, gridmap_list, name_list, gt_list, loc_list, x_list, y_list, a_list

    # get all file path
    birdview_train_fn = np.array(glob.glob(r'' + source_dir + '/train/images/*.' + image_pattern))
    birdview_test_fn = np.array(glob.glob(r'' + source_dir + '/test/images/*.' + image_pattern))
    gridmap_train_fn = np.array(glob.glob(r'' + source_dir + '/train/gridmaps/*.' + gridmap_pattern))
    gridmap_test_fn = np.array(glob.glob(r'' + source_dir + '/test/gridmaps/*.' + gridmap_pattern))
    shifted_train_subdirs = np.array(glob.glob(r'' + source_dir + '/train/shifted_images/*'))
    shifted_test_subdirs = np.array(glob.glob(r'' + source_dir + '/test/shifted_images/*'))

    # sort file paths according to name, in order to align image and gridmap
    birdview_train_fn = sorted(birdview_train_fn, key=file_index)
    gridmap_train_fn = sorted(gridmap_train_fn, key=file_index)
    birdview_test_fn = sorted(birdview_test_fn, key=file_index)
    gridmap_test_fn = sorted(gridmap_test_fn, key=file_index)
    shifted_test_subdirs = sorted(shifted_test_subdirs, key=file_index)
    shifted_train_subdirs = sorted(shifted_train_subdirs, key=file_index)

    image_test_list, gridmap_test_list, name_test_list, gt_test_list, loc_test_list, \
    x_test_list, y_test_list, a_test_list = make_list(birdview_test_fn,
                                                      gridmap_test_fn,
                                                      shifted_test_subdirs,
                                                      screen='test')
    if not only_test:
        image_train_list, gridmap_train_list, name_train_list, gt_train_list, loc_train_list, \
        x_train_list, y_train_list, a_train_list = make_list(birdview_train_fn,
                                                             gridmap_train_fn,
                                                             shifted_train_subdirs,
                                                             screen='train')
    else:
        image_train_list, gridmap_train_list, name_train_list, gt_train_list, loc_train_list, \
        x_train_list, y_train_list, a_train_list = None, None, None, None, None, None, None, None

    if save_list:
        gt_train_series = pd.Series(dtype='object')
        loc_train_series = pd.Series(dtype='object')
        gt_test_series = pd.Series(dtype='object')
        loc_test_series = pd.Series(dtype='object')
        for i in tqdm(range(gt_train_list.shape[0]), desc='train_series'):
            gt_train_series.at[i] = gt_train_list[i, :]
            loc_train_series.at[i] = loc_train_list[i, :]

        for i in tqdm(range(gt_test_list.shape[0]), desc='test_series'):
            gt_test_series.at[i] = gt_test_list[i, :]
            loc_test_series.at[i] = loc_test_list[i, :]

        train_df = pd.DataFrame(
            {"image": image_train_list, "gridmap": gridmap_train_list, "name": name_train_list,
             "x": x_train_list, "y": y_train_list, "angle": a_train_list, "gt": gt_train_series,
             "loc": loc_train_series})
        test_df = pd.DataFrame(
            {"image": image_test_list, "gridmap": gridmap_test_list, "name": name_test_list,
             "x": x_test_list, "y": y_test_list, "angle": a_test_list, "gt": gt_test_series,
             "loc": loc_test_series})
        train_df.to_pickle(source_dir + '/list_train')
        test_df.to_pickle(source_dir + '/list_test')
    else:
        return image_test_list, gridmap_test_list, name_test_list, gt_test_list, loc_test_list, \
               x_test_list, y_test_list, a_test_list, image_train_list, gridmap_train_list, \
               name_train_list, gt_train_list, loc_train_list, x_train_list, y_train_list, a_train_list


def generate_dataset(source_dir=None, time_gap=300., target_dir=None, image_pattern='png',
                     test_proportion=0., gridmap_pattern='png', grayscale=True, screen=True):
    birdview_dir = source_dir + '/images/'
    gridmap_dir = source_dir + '/gridmaps/'
    shift_image_dir = source_dir + '/shifted_images/'

    # get file names of birdview images
    birdview_fn = np.array(glob.glob(r'' + birdview_dir + '*.' + image_pattern))
    if birdview_fn.size is 0:
        print('Folder is empty.')
        exit(1)
    # shuffle the list
    np.random.shuffle(birdview_fn)
    # get timestamp of birdview images from file name.
    birdview_fn = np.array([os.path.splitext(os.path.basename(x))[0] for x in birdview_fn])
    birdview_ts = np.array([int(x) for x in birdview_fn])

    # get subdir names of shifted images
    shift_image_subdirs = np.array(glob.glob(r'' + shift_image_dir + '*'))
    shift_image_subdirs = [os.path.basename(x) for x in shift_image_subdirs]
    shift_image_ts = [int(x) for x in shift_image_subdirs]

    # get file names of gridmaps
    gridmap_fn = np.array(glob.glob(r'' + gridmap_dir + '*.' + gridmap_pattern))

    # get timestamp of gridmaps from file name.
    gridmap_fn = np.array([os.path.splitext(os.path.basename(x))[0] for x in gridmap_fn])
    gridmap_ts = [int(x) for x in gridmap_fn]
    # match timestamps of gridmap and birdview images.
    image_index = np.array([], dtype=int)
    gridmap_index = np.array([], dtype=int)
    for i in tqdm(range(birdview_ts.shape[0]), desc='Pairing images & gridmaps', disable=not screen):
        time_diff = abs(gridmap_ts - birdview_ts[i])
        if time_diff.min() < time_gap and birdview_ts[i] in shift_image_ts:
            image_index = np.append(image_index, i)
            gridmap_index = np.append(gridmap_index, time_diff.argmin())
    # mask paired files
    gridmap_fn = gridmap_fn[gridmap_index]
    birdview_fn = birdview_fn[image_index]

    # save datas in dataset directory
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir + '/train/images')
    os.makedirs(target_dir + '/train/gridmaps')
    os.makedirs(target_dir + '/train/shifted_images')
    os.makedirs(target_dir + '/test/images')
    os.makedirs(target_dir + '/test/gridmaps')
    os.makedirs(target_dir + '/test/shifted_images')

    _pb_train = partial(pb, name='train', target_dir=target_dir, birdview_fn=birdview_fn, birdview_dir=birdview_dir,
                        gridmap_fn=gridmap_fn, gridmap_dir=gridmap_dir, shift_image_dir=shift_image_dir,
                        grayscale=grayscale, image_pattern=image_pattern, gridmap_pattern=gridmap_pattern)
    _pb_test = partial(pb, name='test', target_dir=target_dir, birdview_fn=birdview_fn, birdview_dir=birdview_dir,
                       gridmap_fn=gridmap_fn, gridmap_dir=gridmap_dir, shift_image_dir=shift_image_dir,
                       grayscale=grayscale, image_pattern=image_pattern, gridmap_pattern=gridmap_pattern)

    test_size = int(test_proportion * len(birdview_fn))
    cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=int(cores / 8)) as p:
        with tqdm(total=test_size, desc="generate test dataset", disable=not screen) as pbar:
            for i, _ in enumerate(p.imap(_pb_test, range(0, test_size))):
                pbar.update()

    with multiprocessing.Pool(processes=int(cores / 8)) as p:
        with tqdm(total=len(birdview_fn) - test_size, desc="generate train dataset", disable=not screen) as pbar:
            for i, _ in enumerate(p.imap(_pb_train, range(test_size, len(birdview_fn)))):
                pbar.update()


def pb(x, name, target_dir, birdview_fn, birdview_dir, gridmap_fn, gridmap_dir,
       shift_image_dir, grayscale, image_pattern, gridmap_pattern):
    os.makedirs(target_dir + '/' + name + '/shifted_images/' + str(x))
    save_images(birdview_fn=birdview_fn[x], birdview_dir=birdview_dir,
                gridmap_fn=gridmap_fn[x], gridmap_dir=gridmap_dir,
                shifted_image_subdir=birdview_fn[x], shifted_image_dir=shift_image_dir,
                target_dir=target_dir + '/' + name, grayscale=grayscale, save_name=str(x),
                image_pattern=image_pattern, gridmap_pattern=gridmap_pattern)


def save_images(birdview_fn, birdview_dir, gridmap_fn, gridmap_dir,
                shifted_image_subdir, shifted_image_dir, target_dir, grayscale,
                image_pattern, gridmap_pattern, save_name):
    if grayscale:
        # save birdview
        image = Image.open(birdview_dir + birdview_fn + '.' + image_pattern).convert('L')
        image.save(target_dir + '/images/' + save_name + '.' + image_pattern)
        # save gridmap
        gridmap = Image.open(gridmap_dir + gridmap_fn + '.' + gridmap_pattern).convert('L')
        gridmap.save(target_dir + '/gridmaps/' + save_name + '.' + gridmap_pattern)
        # save shifted image
        shifted_image_files = glob.glob(shifted_image_dir + shifted_image_subdir + '/*')
        shifted_images = [(Image.open(x).convert('L'), os.path.basename(x))
                          for x in shifted_image_files if image_pattern in os.path.splitext(x)[1]]
        for x, name in shifted_images:
            x.save(target_dir + '/shifted_images/' + save_name + '/' + name)
        additional_info = [x for x in shifted_image_files if image_pattern not in os.path.splitext(x)[1]]
        for x in additional_info:
            shutil.copyfile(x, target_dir + '/shifted_images/' + save_name + '/' + os.path.basename(x))
    else:
        # TODO: just copy file to target dir
        pass


# a function to calculate trainable variables in network
def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        num_params += reduce(lambda x, y: x * y, variable.get_shape())
    print("number of parameters: " + str(num_params))


if __name__ == '__main__':
    # generate_dataset(source_dir='/mrtstorage/users/zhu/aigm_small/data_source',
    #                  target_dir='/mrtstorage/users/zhu/aigm_small/dataset', test_proportion=0.1,
    #                  grayscale=True, time_gap=5e7)
    dataset, test_dataset = task_net_input(source_dir='/mrtstorage/users/zhu/aigm_small/dataset',
                                           image_pattern='png', gridmap_pattern='png', pos_weight=5)
    dataset = dataset.map(load_image(img_pattern='png', img_height=320,
                                     img_width=320, grayscale=True,
                                     augumentation=False),
                          num_parallel_calls=4)
    # test = input.map(load_image(img_pattern='png', img_height=300, img_width=200, grayscale=True))
    # it = test.make_one_shot_iterator()
    # ele = it.get_next()
    # with tf.Session() as sess:
    #     Image.fromarray(sess.run(ele)['image'] * 255.).show()
