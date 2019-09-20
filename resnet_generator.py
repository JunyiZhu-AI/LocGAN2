from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import *
from layers import *
from functools import partial
import numpy as np
import math


def resnet_generator_v1(inputs, regul, is_training, concat_cond, init, channel, en_activation, de_activation,
                        bottleneck, momentum=0.997, epsilon=1e-5):
    _res_building_block = partial(res_building_block, is_training=is_training, regul=regul, init=init, version='v1',
                                  momentum=momentum, epsilon=epsilon)

    def _concat_cond(ein):
        if concat_cond:
            cond = tf.image.resize_bilinear(images=inputs, size=ein.shape.as_list()[1:3])
            ein = tf.concat([ein, cond], axis=-1)
        return ein

    with tf.variable_scope('encoder'):
        inputs = tf.concat([inputs], axis=-1)

        en = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=7, activation=en_activation,
                              strides=1, padding="same", use_bias=True,
                              kernel_initializer=init,
                              kernel_regularizer=regul,
                              bias_regularizer=regul,
                              name="conv7")

        en = tf.layers.max_pooling2d(inputs=en, pool_size=3, strides=2, padding='same')

        for i in range(3):
            en = _res_building_block(inputs=en, filters=16, name='conv7_building_block_' + str(i))

        _concat_cond(ein=en)
        en_2 = _res_building_block(inputs=en, filters=32, stride=2, name='downsample0', up_down='down')
        for i in range(3):
            en_2 = _res_building_block(inputs=en_2, filters=32,
                                       name='downsample0_building_block_' + str(i))

        _concat_cond(ein=en_2)
        en_4 = _res_building_block(inputs=en_2, filters=64, stride=2, name='downsample1', up_down='down')
        for i in range(3):
            en_4 = _res_building_block(inputs=en_4, filters=64,
                                       name='downsample1_building_block_' + str(i))

        _concat_cond(ein=en_4)
        en_8 = _res_building_block(inputs=en_4, filters=128, stride=2, name='downsample2', up_down='down')
        for i in range(3):
            en_8 = _res_building_block(inputs=en_8, filters=128, bottleneck=bottleneck,
                                       name='downsample2_building_block_' + str(i))

        _concat_cond(ein=en_8)
        en_16 = _res_building_block(inputs=en_8, filters=256, stride=2, name='downsample3', up_down='down')
        for i in range(5):
            en_16 = _res_building_block(inputs=en_16, filters=256, bottleneck=bottleneck,
                                        name='downsample3_building_block_' + str(i))

    with tf.variable_scope('decoder'):
        _concat_cond(ein=en_16)
        de_16 = _res_building_block(inputs=en_16, filters=128, stride=2, name='upsample0', up_down='up')
        for i in range(3):
            de_16 = _res_building_block(inputs=de_16, filters=128, bottleneck=bottleneck,
                                        name='upsample1_building_block_' + str(i))

        _concat_cond(ein=de_16)
        de_8 = _res_building_block(inputs=de_16, filters=64, stride=2, name='upsample1', up_down='up')
        for i in range(3):
            de_8 = _res_building_block(inputs=de_8, filters=64,
                                       name='upsample2_building_block_' + str(i))

        _concat_cond(ein=de_8)
        de_4 = _res_building_block(inputs=de_8, filters=32, stride=2, name='upsample2', up_down='up')
        for i in range(3):
            de_4 = _res_building_block(inputs=de_4, filters=32,
                                       name='upsample3_building_block_' + str(i))

        _concat_cond(ein=de_4)
        de_2 = _res_building_block(inputs=de_4, filters=16, stride=2, name='upsample3', up_down='up')
        for i in range(3):
            de_2 = _res_building_block(inputs=de_2, filters=16,
                                       name='upsample4_building_block_' + str(i))

        _concat_cond(ein=de_2)
        de = _res_building_block(inputs=de_2, filters=8, stride=2, name='upsample4', up_down='up')
        for i in range(3):
            de = _res_building_block(inputs=de, filters=8,
                                     name='upsample5_building_block_' + str(i))

        de = tf.layers.conv2d(inputs=de, filters=channel, kernel_size=1, activation=de_activation,
                              strides=1, padding="same", use_bias=True,
                              kernel_initializer=init,
                              kernel_regularizer=regul,
                              bias_regularizer=regul,
                              name="conv1")

        return de


def resnet_generator_v2(inputs, regul, is_training, concat_cond, init, channel, en_activation, de_activation,
                        bottleneck, momentum=0.997, epsilon=1e-5, reuse=False):
    '''
    build up a resenet version 2 using the resnet building block.
    '''

    _res_building_block = partial(res_building_block, is_training=is_training, regul=regul, init=init, version='v2',
                                  momentum=momentum, epsilon=epsilon)

    def _concat_cond(ein):
        '''
        concatenate the inputs to the intermiediate layers.
        '''
        cond = None
        if concat_cond:
            cond = tf.image.resize_nearest_neighbor(images=inputs, size=ein.shape.as_list()[1:3])
            ein = tf.concat([ein, cond], axis=-1)
        return ein, cond

    with tf.variable_scope('encoder', reuse=reuse):
        distance_feature = tf.ones_like(inputs, name='distance_feature')
        distance_feature = distance_feature * make_cone(shape=inputs.shape.as_list()[1:3])
        inputs = tf.concat([inputs, distance_feature], axis=-1)

        en = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=7, activation=en_activation,
                              strides=1, padding="same", use_bias=True,
                              kernel_initializer=init,
                              kernel_regularizer=regul,
                              bias_regularizer=regul,
                              name="conv7")
        if not reuse:
            var_list = tf.trainable_variables("generator/encoder/conv7")
            tf.summary.histogram(name="encoder/conv7/weights", values=var_list[0], collections=['train'])
            tf.summary.histogram(name="encoder/conv7/bias", values=var_list[1], collections=['train'])
            tf.summary.histogram(name="encoder/conv7/output", values=en, collections=['train'])

        en_2 = tf.layers.max_pooling2d(inputs=en, pool_size=3, strides=2, padding='same')

        en_2, cond_2 = _concat_cond(ein=en_2)
        en_4 = _res_building_block(inputs=en_2, filters=32, stride=2, name='downsample0', up_down='down')
        for i in range(1):
            en_4 = _res_building_block(inputs=en_4, filters=32,
                                       name='downsample0_building_block_' + str(i))

        en_4, cond_4 = _concat_cond(ein=en_4)
        en_8 = _res_building_block(inputs=en_4, filters=64, stride=2, name='downsample1', up_down='down')
        for i in range(1):
            en_8 = _res_building_block(inputs=en_8, filters=64,
                                       name='downsample1_building_block_' + str(i))

        en_8, cond_8 = _concat_cond(ein=en_8)
        en_16 = _res_building_block(inputs=en_8, filters=128, stride=2, name='downsample2', up_down='down')
        for i in range(3):
            en_16 = _res_building_block(inputs=en_16, filters=128, bottleneck=bottleneck,
                                        name='downsample2_building_block_' + str(i))

        en_16, cond_16 = _concat_cond(ein=en_16)
        en_32 = _res_building_block(inputs=en_16, filters=256, stride=2, name='downsample3', up_down='down')
        for i in range(5):
            en_32 = _res_building_block(inputs=en_32, filters=256, bottleneck=bottleneck,
                                        name='downsample3_building_block_' + str(i))

    with tf.variable_scope('decoder', reuse=reuse):
        de_16 = _res_building_block(inputs=en_32, filters=128, stride=2, name='upsample0', up_down='up')
        for i in range(1):
            de_16 = _res_building_block(inputs=de_16, filters=128, bottleneck=bottleneck,
                                        name='upsample1_building_block_' + str(i))

        if concat_cond:
            de_16 = tf.concat([de_16, cond_16], axis=-1)
        de_8 = _res_building_block(inputs=de_16, filters=64, stride=2, name='upsample1', up_down='up')
        for i in range(1):
            de_8 = _res_building_block(inputs=de_8, filters=64,
                                       name='upsample2_building_block_' + str(i))

        if concat_cond:
            de_8 = tf.concat([de_8, cond_8], axis=-1)
        de_4 = _res_building_block(inputs=de_8, filters=32, stride=2, name='upsample2', up_down='up')
        for i in range(1):
            de_4 = _res_building_block(inputs=de_4, filters=32,
                                       name='upsample3_building_block_' + str(i))

        if concat_cond:
            de_4 = tf.concat([de_4, cond_4], axis=-1)
        de_2 = _res_building_block(inputs=de_4, filters=16, stride=2, name='upsample3', up_down='up')
        for i in range(1):
            de_2 = _res_building_block(inputs=de_2, filters=16,
                                       name='upsample4_building_block_' + str(i))

        de = _res_building_block(inputs=de_2, filters=8, stride=2, name='upsample4', up_down='up')
        for i in range(1):
            de = _res_building_block(inputs=de, filters=8,
                                     name='upsample5_building_block_' + str(i))

        de = tf.layers.batch_normalization(inputs=de, training=is_training,
                                           momentum=momentum, epsilon=epsilon)
        de = tf.nn.relu(de)
        de = tf.layers.conv2d(inputs=de, filters=channel, kernel_size=7, activation=de_activation,
                              strides=1, padding="same", use_bias=True,
                              kernel_initializer=init,
                              kernel_regularizer=regul,
                              bias_regularizer=regul,
                              name="conv7")

        if not reuse:
            var_list = tf.trainable_variables("generator/decoder/conv7")
            tf.summary.histogram(name="decoder/conv7/weights", values=var_list[0], collections=['train'])
            tf.summary.histogram(name="decoder/conv7/bias", values=var_list[1], collections=['train'])
            tf.summary.histogram(name="decoder/conv7/output", values=de, collections=['train'])

        return de


def make_cone(shape):
    '''
    create the distance map
    '''
    mat = np.zeros(shape=shape, dtype=np.float32)
    rows, cols = shape
    for i in range(rows):
        for j in range(cols):
            mat[i, j] = -math.sqrt((i-rows/2.+0.5)**2 + (j-cols/2.+0.5)**2)
    mat = mat - mat.min()
    mat = mat / np.linalg.norm(mat)
    mat = np.expand_dims(mat, axis=0)
    mat = np.expand_dims(mat, axis=-1)
    return mat
