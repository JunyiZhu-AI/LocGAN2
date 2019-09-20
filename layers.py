from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# def dense_norm_layer(inputs, units, name, with_norm, reuse=False, stddev=0.02, activation=tf.nn.relu,
#                      norm=None):
#     # fully connected layer with batch normalization
#     densebn = tf.layers.dense(inputs=inputs, units=units, use_bias=True, name=name, reuse=reuse,
#                               kernel_initializer=tf.random_normal_initializer(stddev=stddev))
#     if with_norm:
#         densebn = norm(inputs=densebn, name=name + '_norm')
#     return activation(densebn)
#
#
# def fracconv_bn_layer(inputs, shape, is_training, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, momentum=0.9,
#                       epsilon=1e-5, activation=tf.nn.relu, with_bn=True):
#     # fractionally convolution with batch normalization
#     w = tf.get_variable('kernel', [k_h, k_w, shape[-1], inputs.shape[-1]],
#                         initializer=tf.random_normal_initializer(stddev=stddev))
#     fracconv = tf.nn.conv2d_transpose(inputs, w, output_shape=shape, strides=[1, s_h, s_w, 1])
#     # add bias
#     bias = tf.get_variable('bias', shape[-1], initializer=tf.constant_initializer(0.0))
#     fracconv = tf.nn.bias_add(fracconv, bias=bias)
#     if with_bn:
#         fracconv = tf.layers.batch_normalization(inputs=fracconv, momentum=momentum, epsilon=epsilon,
#                                                  scale=True, training=is_training)
#     return activation(fracconv)
#
#
# def conv_bn_layer(inputs, feature_deep, is_training, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, momentum=0.9,
#                   epsilon=1e-5, activation=tf.nn.relu, with_bn=True, norm=tf.layers.batch_normalization):
#     w = tf.get_variable('w', [k_w, k_h, inputs.shape[-1], feature_deep],
#                         initializer=tf.truncated_normal_initializer(stddev=stddev))
#     conv = tf.nn.conv2d(inputs, w, strides=[1, s_h, s_w, 1], padding='SAME')
#     bias = tf.get_variable('bias', feature_deep, initializer=tf.constant_initializer(0.0))
#     conv = tf.nn.bias_add(conv, bias=bias)
#     if with_bn:
#         if norm == 'instance':
#             conv = tf.contrib.layers.instance_norm(inputs=conv, epsilon=epsilon)
#         else:
#             conv = norm(inputs=conv, momentum=momentum, epsilon=epsilon,
#                         scale=True, training=is_training)
#     return activation(conv)


def res_building_block(inputs, is_training, regul, filters, version, name, init,
                       momentum, epsilon,
                       up_down=None, bottleneck=False, stride=1, reuse=False):
    '''
    resnet building block. Including architecture version 1 and version 2.
    '''
    assert version == 'v1' or version == 'v2', "unknown verison, valid are v1 or v2"

    with tf.variable_scope(name, reuse=reuse):
        shortcut = inputs
        if stride > 1:
            # projection shortcut
            if up_down == 'down':
                shortcut = tf.layers.conv2d(inputs=shortcut, filters=filters, kernel_size=1, strides=stride,
                                            padding='same',
                                            kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)
            else:
                shortcut = tf.layers.conv2d_transpose(inputs=shortcut, filters=filters, kernel_size=1, strides=stride,
                                                      padding='same', kernel_regularizer=regul, bias_regularizer=regul,
                                                      kernel_initializer=init)

        if version == 'v2':
            inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                   momentum=momentum, epsilon=epsilon)
            inputs = tf.nn.relu(inputs)
            if bottleneck:
                inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=1, strides=1, padding='same',
                                          kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)
                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)
                inputs = tf.nn.relu(inputs)

                if up_down == 'up':
                    inputs = tf.layers.conv2d_transpose(inputs=inputs, filters=64, kernel_size=3, strides=stride,
                                                        padding='same', kernel_regularizer=regul,
                                                        bias_regularizer=regul, kernel_initializer=init)
                else:
                    inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=stride, padding='same',
                                              kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)

                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)
                inputs = tf.nn.relu(inputs)

                inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=1, strides=1, padding='same',
                                          kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)
            else:
                if up_down == 'up':
                    inputs = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=3,
                                                        strides=stride, padding='same', kernel_regularizer=regul,
                                                        bias_regularizer=regul, kernel_initializer=init)
                else:
                    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3,
                                              strides=stride, padding='same', kernel_regularizer=regul,
                                              bias_regularizer=regul, kernel_initializer=init)
                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)
                inputs = tf.nn.relu(inputs)

                inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, strides=1, padding='same',
                                          kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)

            if not reuse:
                tf.summary.histogram(name=name + '_output', values=inputs, collections=['train'])
            return inputs + shortcut

        if version == 'v1':
            if stride > 1:
                shortcut = tf.layers.batch_normalization(inputs=shortcut, training=is_training,
                                                         momentum=momentum, epsilon=epsilon)
            if bottleneck:
                inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=1, strides=1, padding='same',
                                          kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)
                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)
                inputs = tf.nn.relu(inputs)

                if up_down == 'up':
                    inputs = tf.layers.conv2d_transpose(inputs=inputs, filters=64, kernel_size=3, strides=stride,
                                                        padding='same', kernel_regularizer=regul,
                                                        bias_regularizer=regul, kernel_initializer=init)
                else:
                    inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=stride, padding='same',
                                              kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)
                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)
                inputs = tf.nn.relu(inputs)

                inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=1, strides=1, padding='same',
                                          kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)
                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)
            else:
                if up_down == 'up':
                    inputs = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=3, strides=stride,
                                                        padding='same', kernel_regularizer=regul,
                                                        bias_regularizer=regul,
                                                        kernel_initializer=init)
                else:
                    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, strides=stride,
                                              padding='same', kernel_regularizer=regul, bias_regularizer=regul,
                                              kernel_initializer=init)
                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)
                inputs = tf.nn.relu(inputs)

                inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, strides=1, padding='same',
                                          kernel_regularizer=regul, bias_regularizer=regul, kernel_initializer=init)
                inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training,
                                                       momentum=momentum, epsilon=epsilon)

            if not reuse:
                tf.summary.histogram(name=name + '_output', values=inputs, collections=['train'])
            inputs += shortcut
            inputs = tf.nn.relu(inputs)

            return inputs

