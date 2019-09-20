from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from functools import reduce


def gridmap2gridmap_loss(gridmap_, gridmap):
    '''
    pixel-wise loss function computes based on output of sigmoid activation function using cross entropy.
    '''
    with tf.variable_scope("g2g_loss"):
        diff = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gridmap,
                                                                                    logits=gridmap_),
                                            axis=[1, 2, 3]))
        regular_loss = tf.losses.get_regularization_loss(scope='generator')
        tf.summary.scalar(name="pix_loss", tensor=diff, collections=['train', 'test'])
    return tf.math.add_n([diff, regular_loss], name='loss')


def discriminator_loss(logits, real=True):
    noise = tf.random_uniform(shape=[], minval=0, maxval=0.1, dtype=tf.float32, name="soft_label_noise")
    if real:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits)-noise,
                                                       logits=logits, name='error_real')
    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits)+noise,
                                                       logits=logits, name='error_fake')
    return loss


def task_net_loss(logits, data, softlabel, label_noise, usel1, pos_weight):
    # loss_mult = {}
    label = {'x': tf.one_hot(indices=tf.dtypes.cast(tf.math.ceil(data['x'] / 0.5) + 19, dtype=tf.int32), depth=40),
             'y': tf.one_hot(indices=tf.dtypes.cast(tf.math.ceil(data['y'] / 0.5) + 19, dtype=tf.int32), depth=40),
             'a': tf.one_hot(indices=tf.dtypes.cast(tf.math.ceil(data['a'] / 2) + 4, dtype=tf.int32), depth=10)}
    # indices = tf.math.argmax(label[k], axis=-1)
    # conditions = tf.greater_equal(indices, 10)
    # loss_mult[k] = tf.map_fn(lambda x: tf.where(x, 1./pos_weight, 1.), conditions, dtype=tf.float32)
    if softlabel:
        softlabel_filter = tf.constant([0.1, 0.3, 1, 0.3, 0.1], dtype=tf.float32)
        if label_noise:
            softlabel_filter = tf.map_fn(fn=lambda x: x+tf.random.uniform(shape=[], minval=-0.1, maxval=0.1,
                                                                          dtype=tf.float32),
                                         elems=softlabel_filter, dtype=tf.float32)
        softlabel_filter = softlabel_filter / tf.reduce_sum(softlabel_filter, axis=-1)
        for k, v in label.items():
            v = tf.nn.conv1d(value=tf.reshape(tensor=v, shape=[-1, v.shape.as_list()[-1], 1]),
                             filters=tf.reshape(tensor=softlabel_filter,
                                                shape=[softlabel_filter.shape.as_list()[-1], 1, 1]),
                             stride=1, padding='SAME')
            label[k] = tf.squeeze(input=v, axis=-1)

    with tf.name_scope('task_net_loss'):
        loss_dict = {}
        for k, v in label.items():
            precision = tf.math.equal(tf.argmax(input=logits[k], axis=-1), tf.argmax(input=v, axis=-1))
            precision = tf.to_float(precision)
            tf.summary.scalar(name='precision_'+k, tensor=tf.reduce_mean(precision), collections=['train', 'test'])
            distance = tf.abs(tf.argmax(input=logits[k], axis=-1) - tf.argmax(input=v, axis=-1))
            distance = tf.to_float(distance)
            tf.summary.scalar(name='distance_'+k, tensor=tf.reduce_mean(distance), collections=['train', 'test'])
            bad_estimation = tf.to_float((tf.argmax(input=logits[k], axis=-1) - tf.argmax(input=v, axis=-1)) > 3)
            tf.summary.scalar(name='bad_estimation_'+k,
                              tensor=tf.reduce_mean(bad_estimation), collections=['train', 'test'])

            if not usel1:
                # loss_dict[k] = tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(labels=v, logits=logits[k]),
                #                            loss_mult[k])
                loss_dict[k] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=v, logits=logits[k]))
                tf.summary.scalar(name='cross_entropy_'+k, tensor=loss_dict[k],
                                  collections=['train', 'test'])
            else:
                # loss_dict[k] = tf.multiply(tf.reduce_mean(tf.math.abs(v - logits[k]), axis=-1), loss_mult[k])
                loss_dict[k] = tf.reduce_mean(tf.reduce_mean(tf.math.abs(v - logits[k]), axis=-1))
                tf.summary.scalar(name='l1_loss_'+k, tensor=loss_dict[k],
                                  collections=['train', 'test'])

        regular_loss = tf.losses.get_regularization_loss(scope='g_task_net')
        tf.summary.scalar(name="gtn_l2_norm", tensor=regular_loss, collections=['train', 'test'])

    return reduce(lambda x, y: x+y, loss_dict.values())



