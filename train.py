import numpy as np
import tensorflow as tf
from tensorflow import flags, logging, app
from tensorflow.contrib import slim
from pickle import load as pickle_load
from os.path import join as path_join

from read_data import DataTFReader
from constants import NUM_TRAIN_IMAGES, NUM_CLASSES, DataPipeline
from constants import ConvFilterShape, compute_accuracy
from constants import TRAIN_TF_DATA_FILE_NAME, VALIDATION_PICKLE_DATA_FILE_NAME
from constants import TRAIN_VAL_TF_DATA_FILE_NAME
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, IMAGE_SIZE

from linear_model import LogisticRegression

from slim.nets.inception import inception_v4_arg_scope, inception_v4

from functools import reduce
from operator import mul


FLAGS = flags.FLAGS


def tr_log_reg_fn(input_val, **kwargs):
    with tf.name_scope('reshape'):
        return tf.reshape(tf.cast(input_val, tf.float32), [-1, IMAGE_SIZE])


def create_conv_layer(images, filter_shape, strides, name, regularization=True):
    """
    :param images - images or feature maps
    :param filter_shape: namedtuple([filter_height, filter_width, in_channels, out_channels])
        or a list in the form [filter_height, filter_width, in_channels, out_channels]
    :param strides
    :param regularization: Whether to impose regularization or not.
    :param name: An optional name of the variable.
    :return: Variable corresponding to the kernel map or template.
    """
    if isinstance(filter_shape, ConvFilterShape):
        filter_shape = [filter_shape.filter_height, filter_shape.filter_width,
                        filter_shape.in_channels, filter_shape.out_channels]

    bias_shape = filter_shape[-1]
    with tf.variable_scope(name):
        # filter_height * filter_width * in_channels
        fan_in = reduce(mul, filter_shape[:-1], 1)
        # Impose regularization only once when creating the variable.
        weights = tf.get_variable(
            'weights', shape=filter_shape,
            initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(fan_in)),
            regularizer=tf.identity if regularization else None
        )

        # A small positive initial values to avoid dead neurons.
        biases = tf.get_variable('biases', shape=bias_shape,
                                 initializer=tf.constant_initializer(0.01))

        # When padding is 'SAME':
        # out_height = ceil(float(in_height) / float(strides[1]))
        # out_width = ceil(float(in_width) / float(strides[2]))
        output = tf.add(tf.nn.conv2d(images, weights, strides, 'SAME'), biases)

        return output


def tr_data_conv_fn(images, regularization=True, **kwargs):
    reuse = True if 'reuse' in kwargs and kwargs['reuse'] is True else None

    with tf.variable_scope('Pre-process', values=[images], reuse=reuse):
        # Cast images to float type.
        value = tf.cast(images, tf.float32)
        # Scale the imgs to [-1, +1]
        scaled_value = tf.subtract(tf.scalar_mul(2.0 / 255.0, value), 1.0)

    with tf.variable_scope('ConvNet', values=[scaled_value], reuse=reuse):
        all_strides = []
        # Convolutional layer 1.
        filter1_shape = ConvFilterShape(filter_height=3, filter_width=3,
                                        in_channels=IMAGE_CHANNELS, out_channels=32)
        conv1_strides = [1, 1, 1, 1]
        all_strides.append(conv1_strides)
        conv1 = create_conv_layer(scaled_value, filter1_shape, conv1_strides, 'conv1',
                                  regularization=regularization)

        pool1_strides = [1, 2, 2, 1]
        all_strides.append(pool1_strides)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=pool1_strides,
                               padding='SAME', name='max_pool1')

        activation1 = tf.nn.relu(pool1, name='activation1')

        # Convolutional layer 2.
        filter2_shape = ConvFilterShape(filter_height=3, filter_width=3,
                                        in_channels=32, out_channels=64)
        conv2_strides = [1, 1, 1, 1]
        all_strides.append(conv2_strides)
        conv2 = create_conv_layer(activation1, filter2_shape, conv2_strides, 'conv2',
                                  regularization=regularization)

        pool2_strides = [1, 2, 2, 1]
        all_strides.append(pool2_strides)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=pool2_strides,
                               padding='SAME', name='max_pool2')

        activation2 = tf.nn.relu(pool2, name='activation2')

        # Convolutional layer 3.
        filter3_shape = ConvFilterShape(filter_height=3, filter_width=3,
                                        in_channels=64, out_channels=128)
        conv3_strides = [1, 1, 1, 1]
        all_strides.append(conv3_strides)
        conv3 = create_conv_layer(activation2, filter3_shape, conv3_strides, 'conv3',
                                  regularization=regularization)

        pool3_strides = [1, 2, 2, 1]
        all_strides.append(pool3_strides)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=pool3_strides,
                               padding='SAME', name='max_pool3')

        activation3 = tf.nn.relu(pool3, name='activation3')

        # Compute output size of the convolutional layers
        channels = filter3_shape.out_channels
        # Fully connected layer.
        height = IMAGE_HEIGHT
        width = IMAGE_WIDTH
        for e in all_strides:
            height = np.ceil(float(height) / float(e[1]))
            width = np.ceil(float(width) / float(e[2]))

        conv_out_size = int(height * width * channels)
        if reuse is False:
            logging.info(
                'Convolutional layers output {}-dimensional feature.'.format(conv_out_size))

        # Flatten the feature maps
        output = tf.reshape(activation3, [-1, conv_out_size])

        out_size = 1536
        with tf.variable_scope('fc'):
            weights = tf.get_variable(
                'weights', shape=[conv_out_size, out_size],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(conv_out_size)),
                regularizer=tf.identity if regularization else None
            )

            # A small positive initial values to avoid dead neurons.
            biases = tf.get_variable('biases', shape=[out_size],
                                     initializer=tf.constant_initializer(0.01))

            fc = tf.add(tf.matmul(output, weights), biases, name='inner_prod')

        return fc


def transfer_learn_inception_v4(images, **kwargs):
    """
    Fine-tune the inception net
    :param images: The features batch,
        dimension [batch_size x height x width x channels]
    :return: The flattened just before the softmax layer.
    """
    # dropout and batch normalization need to know the phase, training or validation (test).
    # Used for dropout and batch normalization. By default True.
    phase_train_pl = tf.placeholder_with_default(True, [], name='phase_train_pl')
    tf.add_to_collection('phase_train_pl', phase_train_pl)
    # dropout layers
    keep_prob = tf.cond(phase_train_pl, lambda: tf.constant(0.8, name='keep_prob'),
                        lambda: tf.constant(1.0, name='keep_prob'))

    # Do not forget to cast images to float type.
    float_imgs = tf.cast(images, tf.float32)

    # Scale the imgs to [-1, +1]
    # TODO, other pre-process.
    scaled_imgs = tf.subtract(tf.scalar_mul(2.0 / 255.0, float_imgs), 1.0)

    with slim.arg_scope(inception_v4_arg_scope()):
        _, end_points = inception_v4(scaled_imgs, is_training=phase_train_pl)
        net = end_points['PreLogitsFlatten']
        net = tf.nn.dropout(net, keep_prob=keep_prob, name='Dropout')

        return net


def main(unused_argv):
    """
    The training procedure.
    :return:
    """
    reader = DataTFReader(num_classes=NUM_CLASSES)

    with open(FLAGS.validation_data_file, 'rb') as pickle_f:
        val_data, val_labels = pickle_load(pickle_f)

    # TODO, Change Me!
    tr_data_fn = tr_data_conv_fn
    tr_data_paras = {'reshape': True, 'size': 1536}

    train_data_pipeline = DataPipeline(reader=reader,
                                       data_pattern=FLAGS.train_data_pattern,
                                       batch_size=FLAGS.batch_size,
                                       num_threads=FLAGS.num_threads)

    log_reg = LogisticRegression(logdir=path_join(FLAGS.logdir, 'log_reg'))
    log_reg.fit(train_data_pipeline,
                raw_feature_size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                start_new_model=FLAGS.start_new_model,
                tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                validation_set=(val_data, val_labels), validation_fn=compute_accuracy,
                init_learning_rate=0.001, decay_steps=NUM_TRAIN_IMAGES // 2,
                use_pretrain=FLAGS.use_pretrain,
                pretrained_model_dir=FLAGS.pretrained_model_dir,
                pretrained_scope=FLAGS.pretrained_scope)


if __name__ == '__main__':
    logging.set_verbosity(tf.logging.DEBUG)

    flags.DEFINE_string('train_data_pattern', TRAIN_TF_DATA_FILE_NAME,
                        'The Glob pattern to training data tfrecord files.')

    flags.DEFINE_string('train_val_data_file', TRAIN_VAL_TF_DATA_FILE_NAME,
                        'The Glob pattern to fit the linear regression model.')

    flags.DEFINE_integer('batch_size', 64, 'The training batch size.')

    flags.DEFINE_integer('num_threads', 4,
                         'The number of threads to read the tfrecord file.')

    flags.DEFINE_string('validation_data_file', VALIDATION_PICKLE_DATA_FILE_NAME,
                        'The pickle file which stores the validation set.')

    flags.DEFINE_boolean('use_pretrain', False,
                         'Whether to (partially) use pretrained model')

    flags.DEFINE_string('pretrained_model_dir', 'inception_v4_model/',
                        'The pickle file which stores the validation set.')

    flags.DEFINE_string('pretrained_scope', 'InceptionV4',
                        'The variable scope which contains the pretrained variables to store.')

    flags.DEFINE_bool('start_new_model', True, 'Whether to start a new model.')

    flags.DEFINE_string('logdir', '/tmp/inception_v4',
                        'The log dir to log events and checkpoints.')

    app.run()
