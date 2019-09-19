from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import *
from layers import *
from resnet_generator import *
from functools import partial


class Generator:
    def __init__(self, is_training, bottleneck, concat_cond, regular_scale=0.):
        self.regular = tf.contrib.layers.l2_regularizer(regular_scale)
        self.init = tf.contrib.layers.xavier_initializer()
        self.bottleneck = bottleneck
        self.concat_cond = concat_cond
        self.is_training = is_training

    def __call__(self, inputs, version, reuse, epsilon=1e-5, momentum=0.997):
        with tf.variable_scope('generator', reuse=reuse):
            self.step = tf.get_variable(name='step', dtype=tf.int32, trainable=False,
                                        initializer=tf.constant(0, dtype=tf.int32))

            assert version == 'v1' or version == 'v2', "version not known."
            if version == 'v1':
                gridmap_ = resnet_generator_v1(inputs=inputs, regul=self.regular,
                                               channel=1, is_training=self.is_training, en_activation=tf.nn.relu,
                                               de_activation=lambda logits: logits, concat_cond=self.concat_cond,
                                               bottleneck=self.bottleneck, init=self.init)
            else:
                gridmap_ = resnet_generator_v2(inputs=inputs, regul=self.regular,
                                               channel=1, is_training=self.is_training, en_activation=tf.nn.relu,
                                               de_activation=lambda logits: logits, concat_cond=self.concat_cond,
                                               bottleneck=self.bottleneck, init=self.init, reuse=reuse)

        if reuse:
            assert self.variables == tf.trainable_variables('generator'),  "reuse variables failed"
        self.variables = tf.trainable_variables(scope='generator')

        return gridmap_

    def optimizer(self, lr, loss):
        with tf.variable_scope('generator/optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
            with tf.control_dependencies(update_ops):
                optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=1e-5)
                # prevent gradient to be nan, because latent value at beginning can be zero
                grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=self.variables)
                with tf.variable_scope('gradient_clipper'):
                    capped_gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in grads_and_vars]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=self.step)

        return train_op


if __name__ == '__main__':
    G = Generator(is_training=True, bottleneck=True, concat_cond=True)
    inputs1 = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 1])
    inputs2 = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 1])
    ouput1 = G(inputs=inputs1, version='v2')
    output2 = G(inputs=inputs2, version='v1')
