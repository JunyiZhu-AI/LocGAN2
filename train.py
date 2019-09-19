from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import tensorflow as tf
from datetime import datetime
from discriminator import *
from loss_fns import *
from generator import *
from generator_task_net import *
from discriminator_task_net import *
import time
CONFIG = {}


def init_or_restore(sess, saver, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # restore from checkpoint
        print('---------------------')
        print('restore network from PATH: '+ckpt.model_checkpoint_path)
        print('---------------------')
        tf.global_variables_initializer().run()
        saver.restore(sess, ckpt.model_checkpoint_path)

    else:
        print('---------------------')
        print('initial network.')
        print('---------------------')
        tf.global_variables_initializer().run()
        # exist or make directory
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)


def prepare_dataset():
    dataset, test_dataset = image_input(source_dir=CONFIG['dataset_dir'], with_gridmap=True,
                                        image_pattern=CONFIG['image_pattern'],
                                        gridmap_pattern=CONFIG['gridmap_pattern'])
    dataset = dataset.shuffle(5000).repeat()
    test_dataset = test_dataset.shuffle(5000).repeat()
    dataset = dataset.map(load_image(img_pattern=CONFIG['image_pattern'], img_height=CONFIG['img_height'],
                                     img_width=CONFIG['img_width'], grayscale=CONFIG['grayscale'],
                                     augumentation=CONFIG['augumentation']),
                          num_parallel_calls=CONFIG['num_parallel_calls'])
    test_dataset = test_dataset.map(load_image(img_pattern=CONFIG['image_pattern'],
                                               img_height=CONFIG['img_height'],
                                               img_width=CONFIG['img_width'], grayscale=CONFIG['grayscale'],
                                               augumentation=False),
                                    num_parallel_calls=CONFIG['num_parallel_calls'])
    dataset = dataset.batch(CONFIG['batch_size']).prefetch(1)
    test_dataset = test_dataset.batch(CONFIG['batch_size']).prefetch(1)

    handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    one_batch = iterator.get_next()
    train_init = dataset.make_initializable_iterator()
    test_init = test_dataset.make_initializable_iterator()

    return handle, one_batch, train_init, test_init


def prepare_task_net_dataset(load_list):
    dataset, test_dataset = task_net_input(source_dir=CONFIG['dataset_dir'],
                                           image_pattern=CONFIG['image_pattern'],
                                           gridmap_pattern=CONFIG['gridmap_pattern'],
                                           pos_weight=0, only_test=False,
                                           load_list=load_list)
    dataset = dataset.shuffle(5000).repeat()
    test_dataset = test_dataset.shuffle(5000).repeat()
    dataset = dataset.map(load_image(img_pattern=CONFIG['image_pattern'], img_height=CONFIG['img_height'],
                                     img_width=CONFIG['img_width'], grayscale=CONFIG['grayscale'],
                                     augumentation=False),
                          num_parallel_calls=CONFIG['num_parallel_calls'])
    test_dataset = test_dataset.map(load_image(img_pattern=CONFIG['image_pattern'], img_height=CONFIG['img_height'],
                                               img_width=CONFIG['img_width'], grayscale=CONFIG['grayscale'],
                                               augumentation=False),
                                    num_parallel_calls=CONFIG['num_parallel_calls'])
    dataset = dataset.batch(CONFIG['batch_size']).prefetch(2)
    test_dataset = test_dataset.batch(CONFIG['batch_size']).prefetch(2)

    handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    one_batch = iterator.get_next()
    train_init = dataset.make_initializable_iterator()
    test_init = test_dataset.make_initializable_iterator()

    return handle, one_batch, train_init, test_init


def learning_rate_fn(decay, lr, step=None, max_steps=None):
    if decay:
        learning_rate = lr * (((tf.constant(max_steps, dtype=tf.float32) - tf.cast(step, dtype=tf.float32)) /
                              tf.constant(max_steps, dtype=tf.float32)) ** tf.constant(2, dtype=tf.float32)) + lr*1e-6
    else:
        learning_rate = lr

    return learning_rate


def train_generator():
    '''
    generator training function
    '''
    g_config = CONFIG['generator']
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            # prepare dataset
            handle, one_batch, train_init, test_init = prepare_dataset()
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    # initialize a generator
    generator = Generator(bottleneck=g_config['bottleneck'], concat_cond=g_config['concat_cond'],
                          regular_scale=g_config['regular_scale'], is_training=is_training)
    # build generator network
    raw_gridmap = generator(inputs=one_batch['image'], version=g_config['resnet_version'],
                            epsilon=1e-5, momentum=0.99, reuse=False)

    # record images
    with tf.name_scope('images'):
        gridmap_ = tf.nn.sigmoid(x=raw_gridmap, name='fake_gridmap')
        tf.summary.image(name='birdview', tensor=one_batch['image'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap', tensor=one_batch['gridmap'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap_', tensor=gridmap_, max_outputs=4, collections=['train', 'test'])

    # prepare training
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        learning_rate = learning_rate_fn(decay=g_config['lr_decay'], lr=g_config['lr'],
                                         step=generator.step, max_steps=g_config['max_train_steps'])

    loss = gridmap2gridmap_loss(gridmap=one_batch['gridmap'], gridmap_=raw_gridmap,
                                pos_weight=g_config['pos_weight'])
    train_op = generator.optimizer(lr=lr, loss=loss)

    saver = tf.train.Saver(max_to_keep=g_config['ckpt_to_keep'],
                           var_list=tf.global_variables(scope='generator'),
                           name='Saver')
    train_merged = tf.summary.merge_all('train')
    eval_merged = tf.summary.merge_all('test')

    # open session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = CONFIG['gpu_fraction']
    with tf.Session(config=config) as sess:
        summarizer_train = tf.summary.FileWriter(g_config['ckpt_dir'] + "/summaries/train", sess.graph)
        summarizer_test = tf.summary.FileWriter(g_config['ckpt_dir'] + "/summaries/test")

        # initial or restore parameters
        init_or_restore(sess=sess, ckpt_dir=g_config['ckpt_dir'], saver=saver)
        train_handle = sess.run(train_init.string_handle())
        sess.run(train_init.initializer)
        test_handle = sess.run(test_init.string_handle())
        sess.run(test_init.initializer)

        steps_to_save = int(g_config['max_train_steps'] / g_config['ckpt_to_keep'])
        start_time = time.time()

        # check lazy loading
        sess.graph.finalize()

        # print training info
        print("max train steps: {}, remaining train steps: {}, steps to save: {}".format(g_config['max_train_steps'],
                                                                                         g_config['max_train_steps']
                                                                                         - sess.run(generator.step),
                                                                                         steps_to_save))

        try:
            # start to training
            for i in range(1, g_config['max_train_steps'] - sess.run(generator.step) + 1):
                sess.run(train_op, feed_dict={handle: train_handle,
                                              is_training: True,
                                              lr: sess.run(learning_rate)})
                if sess.run(generator.step) % 20 == 0:
                    # print status info every 20 steps
                    current_diff, summary_train = sess.run([loss, train_merged],
                                                           feed_dict={handle: train_handle,
                                                                      is_training: False})

                    duration = time.time() - start_time
                    print("{0} step {1:d}, lr={2:.2e} img_diff={3:.2e} ({4:.2f} sec/batch)"
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sess.run(generator.step),
                                  sess.run(learning_rate), current_diff, duration / i))
                    if sess.run(generator.step) % 100 == 0:
                        # write summary every 100 steps
                        summary_test = sess.run(eval_merged, feed_dict={handle: test_handle,
                                                                        is_training: False})

                        summarizer_train.add_summary(summary_train, sess.run(generator.step))
                        summarizer_test.add_summary(summary_test, sess.run(generator.step))

                if sess.run(generator.step) % steps_to_save == 0:
                    saver.save(sess=sess, save_path=g_config['ckpt_dir'] + '/generator',
                               global_step=sess.run(generator.step),
                               write_meta_graph=False)
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            print(e)
        finally:
            saver.save(sess=sess, save_path=g_config['ckpt_dir'] + '/generator', global_step=sess.run(generator.step),
                       write_meta_graph=False)


def train_discriminator():
    d_config = CONFIG['discriminator']
    g_config = CONFIG['generator']
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            # prepare dataset
            handle, one_batch, train_init, test_init = prepare_dataset()
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    # initialize a generator
    generator = Generator(bottleneck=g_config['bottleneck'], concat_cond=g_config['concat_cond'],
                          regular_scale=g_config['regular_scale'], is_training=False)
    # build generator network
    raw_gridmap = generator(inputs=one_batch['image'], version=g_config['resnet_version'],
                            epsilon=1e-5, momentum=0.997, reuse=False)

    # record images
    with tf.name_scope('images'):
        gridmap_ = tf.nn.sigmoid(x=raw_gridmap, name='fake_gridmap')
        # gridmap_ = tf.multiply(gridmap_, tf.stop_gradient(255./tf.math.reduce_max(gridmap_)))
        tf.summary.image(name='birdview', tensor=one_batch['image'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap', tensor=one_batch['gridmap'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap_', tensor=gridmap_, max_outputs=4, collections=['train', 'test'])
        real = tf.concat([one_batch['image'], one_batch['gridmap']], axis=-1)
        fake = tf.concat([one_batch['image'], gridmap_], axis=-1)

    # initialize a discriminator
    discriminator = Discriminator(is_training=is_training, norm=d_config['norm'],
                                  regular_scale=d_config['regular_scale'])

    # calculate loss
    logit_real = discriminator(inputs=real, reuse=False, epsilon=1e-5, momentum=0.99,
                               dense_units=d_config['dense_unit'])
    logit_fake = discriminator(inputs=fake, reuse=True, epsilon=1e-5, momentum=0.99,
                               dense_units=d_config['dense_unit'])
    error_real = discriminator_loss(logits=logit_real, real=True)
    error_fake = discriminator_loss(logits=logit_fake, real=False)
    d_loss = tf.reduce_mean((error_fake + error_real) / 2, name='d_loss')
    with tf.name_scope('d_losses'):
        tf.summary.scalar('error_real', tf.reduce_mean(error_real), collections=['train', 'test'])
        tf.summary.scalar('error_fake', tf.reduce_mean(error_fake), collections=['train', 'test'])
        tf.summary.scalar('d_loss', d_loss, collections=['train', 'test'])
        regular_loss = tf.losses.get_regularization_loss(scope='discriminator')
        tf.summary.scalar(name="d_l2_Norm", tensor=regular_loss, collections=['train', 'test'])
        d_loss += regular_loss

    # prepare training
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        learning_rate = learning_rate_fn(decay=d_config['lr_decay'], lr=d_config['lr'],
                                         step=discriminator.step, max_steps=d_config['max_train_steps'])

    train_op = discriminator.optimizer(lr=learning_rate, loss=d_loss)

    d_saver = tf.train.Saver(max_to_keep=d_config['ckpt_to_keep'],
                             var_list=tf.global_variables(scope='discriminator'),
                             name='D_Saver')
    g_saver = tf.train.Saver(var_list=tf.global_variables(scope='generator'),
                             name='G_Saver')
    with tf.name_scope('summarizers'):
        train_merged = tf.summary.merge_all('train', scope='^((?!generator).)*$')
        eval_merged = tf.summary.merge_all('test', scope='^((?!generator).)*$')

    # open session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = CONFIG['gpu_fraction']
    with tf.Session(config=config) as sess:
        summarizer_train = tf.summary.FileWriter(d_config['ckpt_dir'] + "/summaries/train", sess.graph)
        summarizer_test = tf.summary.FileWriter(d_config['ckpt_dir'] + "/summaries/test")

        # initial or restore discriminator
        init_or_restore(sess=sess, ckpt_dir=d_config['ckpt_dir'], saver=d_saver)
        # restore generator
        g_ckpt = tf.train.get_checkpoint_state(g_config['ckpt_dir'])
        assert g_ckpt and g_ckpt.model_checkpoint_path, 'generator checkpoint not found!'
        print('---------------------')
        print('restore generator from PATH: '+g_ckpt.model_checkpoint_path)
        print('---------------------')
        g_saver.restore(sess, g_ckpt.model_checkpoint_path)
        train_handle = sess.run(train_init.string_handle())
        sess.run(train_init.initializer)
        test_handle = sess.run(test_init.string_handle())
        sess.run(test_init.initializer)

        steps_to_save = int(d_config['max_train_steps'] / d_config['ckpt_to_keep'])
        start_time = time.time()

        # check lazy loading
        sess.graph.finalize()

        # print training info
        print("max train steps: {}, remaining train steps: {}, steps to save: {}".format(d_config['max_train_steps'],
                                                                                         d_config['max_train_steps']
                                                                                         - sess.run(discriminator.step),
                                                                                         steps_to_save))

        try:
            # start to training
            for i in range(1, d_config['max_train_steps'] - sess.run(discriminator.step) + 1):
                sess.run(train_op, feed_dict={handle: train_handle,
                                              is_training: True,
                                              lr: sess.run(learning_rate)})
                if sess.run(discriminator.step) % 20 == 0:
                    # print status info every 20 steps
                    current_diff, current_real, current_fake, summary_train = sess.run([d_loss, error_real,
                                                                                        error_fake, train_merged],
                                                                                       feed_dict={handle: train_handle,
                                                                                                  is_training: False})

                    duration = time.time() - start_time
                    print("{0} step {1:d}, lr={2:.1e} loss={3:.2e} error_real={4:.2e} "
                          "error_fake={5:.2e} ({6:.2f} sec/batch)".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                          sess.run(discriminator.step),
                                                                          sess.run(learning_rate), current_diff,
                                                                          current_real.mean(), current_fake.mean(),
                                                                          duration / i))
                    if sess.run(discriminator.step) % 100 == 0:
                        # write summary every 100 steps
                        summary_test = sess.run(eval_merged, feed_dict={handle: test_handle,
                                                                        is_training: False})
                        summarizer_train.add_summary(summary_train, sess.run(discriminator.step))
                        summarizer_test.add_summary(summary_test, sess.run(discriminator.step))

                if sess.run(discriminator.step) % steps_to_save == 0:
                    d_saver.save(sess=sess, save_path=d_config['ckpt_dir'] + '/discriminator',
                                 global_step=sess.run(discriminator.step),
                                 write_meta_graph=False)
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            print(e)
        finally:
            d_saver.save(sess=sess, save_path=d_config['ckpt_dir'] + '/discriminator',
                         global_step=sess.run(discriminator.step),
                         write_meta_graph=False)


def train_gan():
    d_config = CONFIG['discriminator']
    g_config = CONFIG['generator']
    gan_config = CONFIG['gan']
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            # prepare dataset
            handle, one_batch, train_init, test_init = prepare_dataset()
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    # initialize a generator
    generator = Generator(bottleneck=g_config['bottleneck'], concat_cond=g_config['concat_cond'],
                          regular_scale=g_config['regular_scale'], is_training=is_training)
    # build generator network
    raw_gridmap = generator(inputs=one_batch['image'], version=g_config['resnet_version'],
                            epsilon=1e-5, momentum=0.99, reuse=False)

    # record images
    with tf.name_scope('images'):
        gridmap_ = tf.nn.sigmoid(x=raw_gridmap, name='fake_gridmap')
        # gridmap_ = tf.multiply(gridmap_, tf.stop_gradient(255./tf.math.reduce_max(gridmap_)))
        tf.summary.image(name='birdview', tensor=one_batch['image'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap', tensor=one_batch['gridmap'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap_', tensor=gridmap_, max_outputs=4, collections=['train', 'test'])
        real = tf.concat([one_batch['image'], one_batch['gridmap']], axis=-1)
        fake = tf.concat([one_batch['image'], gridmap_], axis=-1)

    # initialize a discriminator
    discriminator = Discriminator(is_training=is_training, norm=d_config['norm'],
                                  regular_scale=d_config['regular_scale'])

    # calculate losses
    # discriminator loss
    logit_real = discriminator(inputs=real, reuse=False, epsilon=1e-5, momentum=0.99,
                               dense_units=d_config['dense_unit'])
    logit_fake = discriminator(inputs=fake, reuse=True, epsilon=1e-5, momentum=0.99,
                               dense_units=d_config['dense_unit'])
    error_real = discriminator_loss(logits=logit_real, real=True)
    error_fake = discriminator_loss(logits=logit_fake, real=False)
    d_loss = tf.reduce_mean((error_fake + error_real) / 2, name='d_loss')
    # generator loss
    g_loss = tf.reduce_mean(discriminator_loss(logit_fake, real=True), name='g_loss')
    image_diff = gridmap2gridmap_loss(gridmap=one_batch['gridmap'], gridmap_=raw_gridmap,
                                      pos_weight=g_config['pos_weight'])
    with tf.name_scope('d_losses'):
        tf.summary.scalar('error_real', tf.reduce_mean(error_real), collections=['train', 'test'])
        tf.summary.scalar('error_fake', tf.reduce_mean(error_fake), collections=['train', 'test'])
        tf.summary.scalar('d_loss', d_loss, collections=['train', 'test'])
        regular_loss = tf.losses.get_regularization_loss(scope='discriminator')
        tf.summary.scalar(name="d_l2_Norm", tensor=regular_loss, collections=['train', 'test'])
        d_loss += regular_loss

    with tf.name_scope('g_losses'):
        tf.summary.scalar('gd_loss', g_loss, collections=['train', 'test'])
        regular_loss = tf.losses.get_regularization_loss(scope='generator')
        tf.summary.scalar(name="g_l2_Norm", tensor=regular_loss, collections=['train', 'test'])

    g_loss = gan_config['d_loss_weight']*g_loss + image_diff

    # prepare training
    d_train_op = discriminator.optimizer(lr=d_config['lr'], loss=d_loss)
    g_train_op = generator.optimizer(lr=g_config['lr'], loss=g_loss)

    d_saver = tf.train.Saver(max_to_keep=gan_config['ckpt_to_keep'],
                             var_list=tf.global_variables(scope='discriminator'),
                             name='D_Saver')
    g_saver = tf.train.Saver(max_to_keep=gan_config['ckpt_to_keep'],
                             var_list=tf.global_variables(scope='generator'),
                             name='G_Saver')

    with tf.name_scope('summarizers'):
        train_merged = tf.summary.merge_all('train')
        eval_merged = tf.summary.merge_all('test')

    # open session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = CONFIG['gpu_fraction']
    with tf.Session(config=config) as sess:
        summarizer_train = tf.summary.FileWriter(gan_config['ckpt_dir'] + "/summaries/train", sess.graph)
        summarizer_test = tf.summary.FileWriter(gan_config['ckpt_dir'] + "/summaries/test")

        if gan_config['initial_gan']:
            d_ckpt_dir = d_config['ckpt_dir']
            g_ckpt_dir = g_config['ckpt_dir']
        else:
            d_ckpt_dir = gan_config['ckpt_dir'] + '/discriminator'
            g_ckpt_dir = gan_config['ckpt_dir'] + '/generator'

        # restore discriminator
        d_ckpt = tf.train.get_checkpoint_state(d_ckpt_dir)
        assert d_ckpt and d_ckpt.model_checkpoint_path, 'discriminator checkpoint not found!'
        print('---------------------')
        print('restore discriminator from PATH: '+d_ckpt.model_checkpoint_path)
        print('---------------------')
        d_saver.restore(sess, d_ckpt.model_checkpoint_path)
        # restore generator
        g_ckpt = tf.train.get_checkpoint_state(g_ckpt_dir)
        assert g_ckpt and g_ckpt.model_checkpoint_path, 'generator checkpoint not found!'
        print('---------------------')
        print('restore generator from PATH: '+g_ckpt.model_checkpoint_path)
        print('---------------------')
        g_saver.restore(sess, g_ckpt.model_checkpoint_path)
        train_handle = sess.run(train_init.string_handle())
        sess.run(train_init.initializer)
        test_handle = sess.run(test_init.string_handle())
        sess.run(test_init.initializer)

        steps_to_save = int(gan_config['max_train_steps'] / gan_config['ckpt_to_keep'])
        start_time = time.time()
        if gan_config['initial_gan']:
            global_step = 0
        else:
            global_step = int(os.path.basename(g_ckpt.model_checkpoint_path).split('-')[1])

        # check lazy loading
        sess.graph.finalize()

        # print training info
        print("max train steps: {}, global steps: {} steps to save: {}".format(gan_config['max_train_steps'],
                                                                               global_step,
                                                                               steps_to_save))

        try:
            # start to training
            for i in range(1, gan_config['max_train_steps'] - global_step + 1):
                sess.run([d_train_op, g_train_op], feed_dict={handle: train_handle,
                                                              is_training: True})
                global_step += 1
                if global_step % 20 == 0:
                    # print status info every 20 steps
                    d, g, image, current_real, current_fake, summary_train = sess.run([d_loss, g_loss,
                                                                                       image_diff, error_real,
                                                                                       error_fake, train_merged],
                                                                                      feed_dict={handle: train_handle,
                                                                                                 is_training: False})

                    duration = time.time() - start_time
                    print("{0} step {1:d}, d_loss={2:.2e} g_loss={3:.2e} image_diff={4:.2e} error_real={5:.2e} "
                          "error_fake={6:.2e} ({7:.2f} sec/batch)".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                          global_step, d, g, image,
                                                                          current_real.mean(), current_fake.mean(),
                                                                          duration / i))
                    if global_step % 100 == 0:
                        # write summary every 100 steps
                        summary_test = sess.run(eval_merged, feed_dict={handle: test_handle,
                                                                        is_training: False})
                        summarizer_train.add_summary(summary_train, global_step)
                        summarizer_test.add_summary(summary_test, global_step)

                if global_step % steps_to_save == 0:
                    d_saver.save(sess=sess, save_path=gan_config['ckpt_dir'] + '/discriminator/',
                                 global_step=global_step,
                                 write_meta_graph=False)
                    g_saver.save(sess=sess, save_path=gan_config['ckpt_dir'] + '/generator/',
                                 global_step=global_step,
                                 write_meta_graph=False)
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            print(e)
        finally:
            d_saver.save(sess=sess, save_path=gan_config['ckpt_dir'] + '/discriminator/', global_step=global_step,
                         write_meta_graph=False)
            g_saver.save(sess=sess, save_path=gan_config['ckpt_dir'] + '/generator/', global_step=global_step,
                         write_meta_graph=False)


def train_discriminator_task_net():
    d_config = CONFIG['discriminator']
    dt_config = CONFIG['discriminator_task_net']
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            # prepare dataset
            handle, one_batch, train_init, test_init = prepare_task_net_dataset(load_list=True)
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    # initialize a discriminator
    discriminator = Discriminator(is_training=False, norm=d_config['norm'],
                                  regular_scale=d_config['regular_scale'])

    # record images
    with tf.name_scope('images'):
        tf.summary.image(name='birdview', tensor=one_batch['image'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap', tensor=one_batch['gridmap'], max_outputs=4, collections=['train', 'test'])

    # build discriminator
    _ = discriminator(inputs=tf.concat([one_batch['image'], one_batch['gridmap']], axis=-1),
                      reuse=False, epsilon=1e-5, momentum=0.99,
                      dense_units=d_config['dense_unit'])
    image_features = tf.get_default_graph().get_tensor_by_name('discriminator/image_features:0')

    # initialize a task net
    dt = D_Task_Net(is_training=is_training, norm=dt_config['norm'], regular_scale=dt_config['regular_scale'])
    logits = dt(inputs=image_features, dense_units=dt_config['dense_units'], epsilon=1e-5, momentum=0.99)

    # prepare training
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        learning_rate = learning_rate_fn(decay=dt_config['lr_decay'], lr=dt_config['lr'],
                                         step=dt.step, max_steps=dt_config['max_train_steps'])

    loss = task_net_loss(logits=logits, label=one_batch['label'], softlabel=dt_config['softlabel'],
                         usel1=dt_config['usel1'], label_noise=dt_config['label_noise'],
                         pos_weight=CONFIG['task_net_pos_weight'])

    train_op = dt.optimizer(lr=lr, loss=loss)

    d_saver = tf.train.Saver(max_to_keep=dt_config['ckpt_to_keep'],
                             var_list=tf.global_variables(scope='discriminator'),
                             name='D_Saver')
    saver = tf.train.Saver(max_to_keep=dt_config['ckpt_to_keep'],
                           var_list=tf.global_variables(scope='d_task_net'),
                           name='Saver')
    train_merged = tf.summary.merge_all('train')
    eval_merged = tf.summary.merge_all('test')

    # open session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = CONFIG['gpu_fraction']
    with tf.Session(config=config) as sess:
        summarizer_train = tf.summary.FileWriter(dt_config['task_net_ckpt_dir'] + "/summaries/train", sess.graph)
        summarizer_test = tf.summary.FileWriter(dt_config['task_net_ckpt_dir'] + "/summaries/test")

        # initial or restore parameters
        init_or_restore(sess=sess, ckpt_dir=dt_config['task_net_ckpt_dir'], saver=saver)
        # restore discriminator
        d_ckpt = tf.train.get_checkpoint_state(dt_config['d_ckpt_dir'])
        assert d_ckpt and d_ckpt.model_checkpoint_path, 'discriminator checkpoint not found!'
        print('---------------------')
        print('restore discriminator from PATH: '+d_ckpt.model_checkpoint_path)
        print('---------------------')
        d_saver.restore(sess, d_ckpt.model_checkpoint_path)

        # initial dataset handles
        train_handle = sess.run(train_init.string_handle())
        sess.run(train_init.initializer)
        test_handle = sess.run(test_init.string_handle())
        sess.run(test_init.initializer)

        steps_to_save = int(dt_config['max_train_steps'] / dt_config['ckpt_to_keep'])
        start_time = time.time()

        # check lazy loading
        sess.graph.finalize()

        # print training info
        print("max train steps: {}, remaining train steps: {}, steps to save: {}".format(dt_config['max_train_steps'],
                                                                                         dt_config['max_train_steps']
                                                                                         - sess.run(dt.step),
                                                                                         steps_to_save))

        try:
            # start to training
            for i in range(1, dt_config['max_train_steps'] - sess.run(dt.step) + 1):
                sess.run(train_op, feed_dict={handle: train_handle,
                                              is_training: True,
                                              lr: sess.run(learning_rate)})
                if sess.run(dt.step) % 20 == 0:
                    # print status info every 20 steps
                    current_diff, summary_train = sess.run([loss, train_merged],
                                                           feed_dict={handle: train_handle,
                                                                      is_training: False})

                    duration = time.time() - start_time
                    print("{0} step {1:d}, lr={2:.2e} loss={3:.2e} ({4:.2f} sec/batch)"
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sess.run(dt.step),
                                  sess.run(learning_rate), current_diff.mean(), duration / i))
                    if sess.run(dt.step) % 100 == 0:
                        # write summary every 100 steps
                        summary_test = sess.run(eval_merged, feed_dict={handle: test_handle,
                                                                        is_training: False})

                        summarizer_train.add_summary(summary_train, sess.run(dt.step))
                        summarizer_test.add_summary(summary_test, sess.run(dt.step))

                if sess.run(dt.step) % steps_to_save == 0:
                    saver.save(sess=sess, save_path=dt_config['task_net_ckpt_dir']+'/',
                               global_step=sess.run(dt.step),
                               write_meta_graph=False)
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            print(e)
        finally:
            saver.save(sess=sess, save_path=dt_config['task_net_ckpt_dir']+'/', global_step=sess.run(dt.step),
                       write_meta_graph=False)


def train_generator_task_net():
    g_config = CONFIG['generator']
    gt_config = CONFIG['generator_task_net']
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            # prepare dataset
            handle, one_batch, train_init, test_init = prepare_task_net_dataset(load_list=gt_config['load_list'])
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    # initialize a generator
    generator = Generator(bottleneck=g_config['bottleneck'], concat_cond=g_config['concat_cond'],
                          regular_scale=g_config['regular_scale'], is_training=False)
    # build generator network
    raw_gridmap = generator(inputs=one_batch['image'], version=g_config['resnet_version'],
                            epsilon=1e-5, momentum=0.99, reuse=False)

    # record images
    with tf.name_scope('images'):
        gridmap_ = tf.nn.sigmoid(x=raw_gridmap)
        gridmap_ = tf.multiply(gridmap_, 1./tf.math.reduce_max(gridmap_))
        tf.summary.image(name='birdview', tensor=one_batch['image'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap', tensor=one_batch['gridmap'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap_', tensor=gridmap_, max_outputs=4, collections=['train', 'test'])

    # initialize a task net
    gt = G_Task_Net(is_training=is_training, norm=gt_config['norm'], regular_scale=gt_config['regular_scale'])
    logits = gt(inputs=tf.concat([gridmap_, one_batch['gridmap']], axis=-1), epsilon=1e-5, momentum=0.99)

    # prepare training
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        learning_rate = learning_rate_fn(decay=gt_config['lr_decay'], lr=gt_config['lr'],
                                         step=gt.step, max_steps=gt_config['max_train_steps'])

    loss = task_net_loss(logits=logits, data=one_batch, softlabel=gt_config['softlabel'],
                         usel1=gt_config['usel1'], label_noise=gt_config['label_noise'],
                         pos_weight=CONFIG['task_net_pos_weight'])
    regular_loss = tf.losses.get_regularization_loss(scope='g_task_net')
    loss += regular_loss

    train_op = gt.optimizer(lr=lr, loss=loss)

    g_saver = tf.train.Saver(max_to_keep=gt_config['ckpt_to_keep'],
                             var_list=tf.global_variables(scope='generator'),
                             name='G_Saver')
    saver = tf.train.Saver(max_to_keep=gt_config['ckpt_to_keep'],
                           var_list=tf.global_variables(scope='g_task_net'),
                           name='GT_Saver')
    train_merged = tf.summary.merge_all('train')
    eval_merged = tf.summary.merge_all('test')

    # open session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = CONFIG['gpu_fraction']
    with tf.Session(config=config) as sess:
        summarizer_train = tf.summary.FileWriter(gt_config['task_net_ckpt_dir'] + "/summaries/train", sess.graph)
        summarizer_test = tf.summary.FileWriter(gt_config['task_net_ckpt_dir'] + "/summaries/test")

        # initial or restore parameters
        init_or_restore(sess=sess, ckpt_dir=gt_config['task_net_ckpt_dir'], saver=saver)
        # restore generator
        g_ckpt = tf.train.get_checkpoint_state(gt_config['g_ckpt_dir'])
        assert g_ckpt and g_ckpt.model_checkpoint_path, 'generator checkpoint not found!'
        print('---------------------')
        print('restore generator from PATH: '+g_ckpt.model_checkpoint_path)
        print('---------------------')
        g_saver.restore(sess, g_ckpt.model_checkpoint_path)

        # initial dataset handels
        train_handle = sess.run(train_init.string_handle())
        sess.run(train_init.initializer)
        test_handle = sess.run(test_init.string_handle())
        sess.run(test_init.initializer)

        steps_to_save = int(gt_config['max_train_steps'] / gt_config['ckpt_to_keep'])
        start_time = time.time()

        # check lazy loading
        sess.graph.finalize()

        # print training info
        print("max train steps: {}, remaining train steps: {}, steps to save: {}".format(gt_config['max_train_steps'],
                                                                                         gt_config['max_train_steps']
                                                                                         - sess.run(gt.step),
                                                                                         steps_to_save))

        try:
            # start to training
            for i in range(1, gt_config['max_train_steps'] - sess.run(gt.step) + 1):
                sess.run(train_op, feed_dict={handle: train_handle,
                                              is_training: True,
                                              lr: sess.run(learning_rate)})
                if sess.run(gt.step) % 20 == 0:
                    # print status info every 20 steps
                    current_diff, summary_train = sess.run([loss, train_merged],
                                                           feed_dict={handle: train_handle,
                                                                      is_training: False})

                    duration = time.time() - start_time
                    print("{0} step {1:d}, lr={2:.2e} loss={3:.2e} ({4:.2f} sec/batch)"
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sess.run(gt.step),
                                  sess.run(learning_rate), current_diff.mean(), duration / i))
                    if sess.run(gt.step) % 100 == 0:
                        # write summary every 100 steps
                        summary_test = sess.run(eval_merged, feed_dict={handle: test_handle,
                                                                        is_training: False})

                        summarizer_train.add_summary(summary_train, sess.run(gt.step))
                        summarizer_test.add_summary(summary_test, sess.run(gt.step))

                if sess.run(gt.step) % steps_to_save == 0:
                    saver.save(sess=sess, save_path=gt_config['task_net_ckpt_dir']+'/',
                               global_step=sess.run(gt.step),
                               write_meta_graph=False)
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            print(e)
        finally:
            saver.save(sess=sess, save_path=gt_config['task_net_ckpt_dir']+'/', global_step=sess.run(gt.step),
                       write_meta_graph=False)


def train_locgan():
    d_config = CONFIG['discriminator']
    g_config = CONFIG['generator']
    gt_config = CONFIG['generator_task_net']
    locgan_config = CONFIG['locgan']
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            # prepare dataset
            handle, one_batch, train_init, test_init = prepare_dataset()
            task_handle, task_one_batch, task_train_init, task_test_init = \
                prepare_task_net_dataset(load_list=gt_config['load_list'])

    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    # initialize a generator
    generator = Generator(bottleneck=g_config['bottleneck'], concat_cond=g_config['concat_cond'],
                          regular_scale=g_config['regular_scale'], is_training=is_training)
    # build generator network
    raw_gridmap = generator(inputs=one_batch['image'], version=g_config['resnet_version'],
                            epsilon=1e-5, momentum=0.99, reuse=False)
    task_raw_gridmap = generator(inputs=task_one_batch['image'], version=g_config['resnet_version'],
                                 epsilon=1e-5, momentum=0.99, reuse=True)

    # record images
    with tf.name_scope('images'):
        gridmap_ = tf.nn.sigmoid(x=raw_gridmap, name='fake_gridmap')
        gridmap_ = tf.multiply(gridmap_, tf.stop_gradient(255./tf.math.reduce_max(gridmap_)))
        task_gridmap_ = tf.nn.sigmoid(x=task_raw_gridmap, name='task_fake_gridmap')
        task_gridmap_ = tf.multiply(task_gridmap_, tf.stop_gradient(255./tf.math.reduce_max(task_gridmap_)))
        tf.summary.image(name='birdview', tensor=one_batch['image'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap', tensor=one_batch['gridmap'], max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='gridmap_', tensor=gridmap_, max_outputs=4, collections=['train', 'test'])
        tf.summary.image(name='task_birdview', tensor=task_one_batch['image'], max_outputs=4,
                         collections=['train', 'test'])
        tf.summary.image(name='task_gridmap', tensor=task_one_batch['gridmap'], max_outputs=4,
                         collections=['train', 'test'])
        tf.summary.image(name='task_gridmap_', tensor=task_gridmap_, max_outputs=4, collections=['train', 'test'])
        real = tf.concat([one_batch['image'], one_batch['gridmap']], axis=-1)
        fake = tf.concat([one_batch['image'], gridmap_], axis=-1)

    # initialize a discriminator
    discriminator = Discriminator(is_training=is_training, norm=d_config['norm'],
                                  regular_scale=d_config['regular_scale'])
    # initialize a task net
    gt = G_Task_Net(is_training=is_training, norm=gt_config['norm'], regular_scale=gt_config['regular_scale'])
    # build generator task net
    logits = gt(inputs=tf.concat([task_gridmap_, task_one_batch['gridmap']], axis=-1), epsilon=1e-5, momentum=0.99)

    # calculate losses
    # discriminator loss
    logit_real = discriminator(inputs=real, reuse=False, epsilon=1e-5, momentum=0.99,
                               dense_units=d_config['dense_unit'])
    logit_fake = discriminator(inputs=fake, reuse=True, epsilon=1e-5, momentum=0.99,
                               dense_units=d_config['dense_unit'])
    error_real = discriminator_loss(logits=logit_real, real=True)
    error_fake = discriminator_loss(logits=logit_fake, real=False)
    d_loss = tf.reduce_mean((error_fake + error_real) / 2, name='d_loss')
    # generator task net loss without regularization
    gtn_raw_loss = task_net_loss(logits=logits, data=task_one_batch, softlabel=gt_config['softlabel'],
                                 usel1=gt_config['usel1'], label_noise=gt_config['label_noise'],
                                 pos_weight=CONFIG['task_net_pos_weight'])
    # generator loss
    gd_loss = tf.reduce_mean(discriminator_loss(logit_fake, real=True), name='g_loss')
    image_diff = gridmap2gridmap_loss(gridmap=one_batch['gridmap'], gridmap_=raw_gridmap,
                                      pos_weight=g_config['pos_weight'])
    g_loss = locgan_config['d_loss_weight']*gd_loss + image_diff + locgan_config['gtn_loss_weight']*gtn_raw_loss

    # generator task net loss with regularization
    regular_loss = tf.losses.get_regularization_loss(scope='g_task_net')
    gtn_loss = regular_loss + gtn_raw_loss

    with tf.name_scope('d_losses'):
        tf.summary.scalar('error_real', tf.reduce_mean(error_real), collections=['train', 'test'])
        tf.summary.scalar('error_fake', tf.reduce_mean(error_fake), collections=['train', 'test'])
        tf.summary.scalar('d_loss', d_loss, collections=['train', 'test'])
        regular_loss = tf.losses.get_regularization_loss(scope='discriminator')
        tf.summary.scalar(name="d_l2_norm", tensor=regular_loss, collections=['train', 'test'])
        d_loss += regular_loss

    with tf.name_scope('g_losses'):
        tf.summary.scalar('gd_loss', gd_loss, collections=['train', 'test'])
        regular_loss = tf.losses.get_regularization_loss(scope='generator')
        tf.summary.scalar(name="g_l2_norm", tensor=regular_loss, collections=['train', 'test'])


    # backpropagation
    # optimize discriminator
    d_train_op = discriminator.optimizer(lr=d_config['lr'], loss=d_loss)
    # optimize generator
    g_train_op = generator.optimizer(lr=g_config['lr'], loss=g_loss)
    # optimize generator task net
    gt_train_op = gt.optimizer(lr=gt_config['lr'], loss=gtn_loss)

    d_saver = tf.train.Saver(max_to_keep=locgan_config['ckpt_to_keep'],
                             var_list=tf.global_variables(scope='discriminator'),
                             name='D_Saver')
    g_saver = tf.train.Saver(max_to_keep=locgan_config['ckpt_to_keep'],
                             var_list=tf.global_variables(scope='generator'),
                             name='G_Saver')
    gt_saver = tf.train.Saver(max_to_keep=locgan_config['ckpt_to_keep'],
                              var_list=tf.global_variables(scope='g_task_net'),
                              name='G_T_Saver')

    with tf.name_scope('summarizers'):
        train_merged = tf.summary.merge_all('train')
        eval_merged = tf.summary.merge_all('test')

    # open session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = CONFIG['gpu_fraction']
    with tf.Session(config=config) as sess:
        summarizer_train = tf.summary.FileWriter(locgan_config['ckpt_dir'] + "/summaries/train", sess.graph)
        summarizer_test = tf.summary.FileWriter(locgan_config['ckpt_dir'] + "/summaries/test")

        d_ckpt_dir = locgan_config['D_dir']
        g_ckpt_dir = locgan_config['G_dir']
        gt_ckpt_dir = locgan_config['GT_dir']
        # restore discriminator
        d_ckpt = tf.train.get_checkpoint_state(d_ckpt_dir)
        assert d_ckpt and d_ckpt.model_checkpoint_path, 'discriminator checkpoint not found!'
        print('---------------------')
        print('restore discriminator from PATH: '+d_ckpt.model_checkpoint_path)
        print('---------------------')
        d_saver.restore(sess, d_ckpt.model_checkpoint_path)
        # restore generator
        g_ckpt = tf.train.get_checkpoint_state(g_ckpt_dir)
        assert g_ckpt and g_ckpt.model_checkpoint_path, 'generator checkpoint not found!'
        print('---------------------')
        print('restore generator from PATH: '+g_ckpt.model_checkpoint_path)
        print('---------------------')
        g_saver.restore(sess, g_ckpt.model_checkpoint_path)
        # restore generator task net
        gt_ckpt = tf.train.get_checkpoint_state(gt_ckpt_dir)
        assert gt_ckpt and gt_ckpt.model_checkpoint_path, 'generator task net checkpoint not found!'
        print('---------------------')
        print('restore generator task net from PATH: '+gt_ckpt.model_checkpoint_path)
        print('---------------------')
        gt_saver.restore(sess, gt_ckpt.model_checkpoint_path)

        train_handle = sess.run(train_init.string_handle())
        sess.run(train_init.initializer)
        test_handle = sess.run(test_init.string_handle())
        sess.run(test_init.initializer)
        task_train_handle = sess.run(task_train_init.string_handle())
        sess.run(task_train_init.initializer)
        task_test_handle = sess.run(task_test_init.string_handle())
        sess.run(task_test_init.initializer)

        steps_to_save = int(locgan_config['max_train_steps'] / locgan_config['ckpt_to_keep'])
        start_time = time.time()
        if locgan_config['initial_gan']:
            global_step = 0
        else:
            global_step = int(os.path.basename(g_ckpt.model_checkpoint_path).split('-')[1])

        # check lazy loading
        sess.graph.finalize()

        # print training info
        print("max train steps: {}, global steps: {} steps to save: {}".format(locgan_config['max_train_steps'],
                                                                               global_step,
                                                                               steps_to_save))

        try:
            # start to training
            for i in range(1, locgan_config['max_train_steps'] - global_step + 1):
                sess.run([d_train_op, g_train_op, gt_train_op], feed_dict={handle: train_handle,
                                                                           task_handle: task_train_handle,
                                                                           is_training: True})
                global_step += 1
                if global_step % 20 == 0:
                    # print status info every 20 steps
                    d, g, image, gtn, summary_train = sess.run([d_loss, g_loss, image_diff, gtn_loss,
                                                                train_merged],
                                                               feed_dict={handle: train_handle,
                                                                          task_handle: task_train_handle,
                                                                          is_training: False})
                    duration = time.time() - start_time
                    print("{0} step {1:d}, d_loss={2:.2e} g_loss={3:.2e} image_diff={4:.2e} gt_loss={5:.2f}"
                          " ({6:.2f} sec/batch)".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                        global_step, d, g, image, gtn, duration / i))
                    if global_step % 100 == 0:
                        # write summary every 100 steps
                        summary_test = sess.run(eval_merged, feed_dict={handle: test_handle,
                                                                        task_handle: task_test_handle,
                                                                        is_training: False})
                        summarizer_train.add_summary(summary_train, global_step)
                        summarizer_test.add_summary(summary_test, global_step)

                if global_step % steps_to_save == 0:
                    d_saver.save(sess=sess, save_path=locgan_config['ckpt_dir'] + '/discriminator/',
                                 global_step=global_step,
                                 write_meta_graph=False)
                    g_saver.save(sess=sess, save_path=locgan_config['ckpt_dir'] + '/generator/',
                                 global_step=global_step,
                                 write_meta_graph=False)
                    gt_saver.save(sess=sess, save_path=locgan_config['ckpt_dir'] + '/gtn/',
                                  global_step=global_step,
                                  write_meta_graph=False)
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            print(e)
        finally:
            d_saver.save(sess=sess, save_path=locgan_config['ckpt_dir'] + '/discriminator/', global_step=global_step,
                         write_meta_graph=False)
            g_saver.save(sess=sess, save_path=locgan_config['ckpt_dir'] + '/generator/', global_step=global_step,
                         write_meta_graph=False)
            gt_saver.save(sess=sess, save_path=locgan_config['ckpt_dir'] + '/gtn/', global_step=global_step,
                          write_meta_graph=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    # add positional argument command
    parser.add_argument('command', help='Command to run',
                        choices=['train_discriminator', 'train_generator', 'train_generator_task_net', 'test_generator',
                                 'train_discriminator_task_net', 'train_gan', 'train_locgan'])
    # add optinal argument config
    parser.add_argument('--config', default='config.yaml', help='path to the config file')
    del argv[0]
    args = parser.parse_args(argv)
    # load config file
    with open(args.config, 'r') as stream:
        global CONFIG
        CONFIG = yaml.load(stream)

    if args.command == 'train_generator':
        train_generator()
    elif args.command == 'train_discriminator':
        train_discriminator()
    elif args.command == 'train_gan':
        train_gan()
    elif args.command == 'train_discriminator_task_net':
        train_discriminator_task_net()
    elif args.command == 'train_generator_task_net':
        train_generator_task_net()
    elif args.command == 'train_locgan':
        train_locgan()
    else:
        print("Unknow command")
        exit(1)


if __name__ == "__main__":
    # tf.logging.set_verbosity(tf.logging.INFO)
    try:
        tf.app.run(main)
    except KeyboardInterrupt:
        print('EXIT')
