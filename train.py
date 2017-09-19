import numpy as np
import tensorflow as tf
from tensorflow import flags, logging, app

from read_data import DataTFReader
from constants import NUM_TRAIN_IMAGES, NUM_CLASSES, DataPipeline, ConvFilterShape
from constants import TRAIN_TF_DATA_FILE_NAME, VALIDATION_TF_DATA_FILE_NAME
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, IMAGE_SIZE

from linear_model import LogisticRegression

from functools import reduce
from operator import mul


FLAGS = flags.FLAGS


def flatten(input_val, **kwargs):
    with tf.name_scope('reshape'):
        return tf.reshape(tf.cast(input_val, tf.float32), [-1, IMAGE_SIZE])


def weight_variable(shape, regularization=False, name='weights'):
    """
    :param shape: namedtuple([filter_height, filter_width, in_channels, out_channels])
        or a list in the form [filter_height, filter_width, in_channels, out_channels]
    :param regularization: Whether to impose regularization or not.
    :param name: An optional name of the variable.
    :return: Variable corresponding to the kernel map or template.
    """
    if isinstance(shape, ConvFilterShape):
        shape = [shape.filter_height, shape.filter_width, shape.in_channels, shape.out_channels]

    # filter_height * filter_width * in_channels
    fan_in = reduce(mul, shape[:-1], 1)

    initial = tf.truncated_normal(shape, stddev=1.0 / np.sqrt(fan_in))

    weights = tf.Variable(initial_value=initial, name=name)
    if regularization:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
        tf.summary.histogram('model/weights', weights)

    return weights


def bias_variable(shape, name='biases'):
    """
    :param shape: [dim]
    :param name: An optional name of the variable.
    :return: Variable corresponding to the kernel map or template.
    """
    if isinstance(shape, ConvFilterShape):
        shape = [shape.out_channels]

    # A small positive initial values to avoid dead neurons.
    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial_value=initial, name=name)


def create_conv_layer(input_val, filter_shape, strides, name):
    with tf.name_scope(name):
        weights = weight_variable(filter_shape, regularization=True)
        biases = bias_variable(filter_shape)

        # When padding is 'SAME':
        # out_height = ceil(float(in_height) / float(strides[1]))
        # out_width = ceil(float(in_width) / float(strides[2]))
        output = tf.add(tf.nn.conv2d(input_val, weights, strides, 'SAME'), biases)

        return output


def tr_data_conv_fn(input_val, **kwargs):
    value = tf.cast(input_val, tf.float32)
    all_strides = []
    # Convolutional layer 1.
    filter1_shape = ConvFilterShape(filter_height=3, filter_width=3,
                                    in_channels=IMAGE_CHANNELS, out_channels=32)
    conv1_strides = [1, 2, 2, 1]
    all_strides.append(conv1_strides)
    conv1 = create_conv_layer(value, filter1_shape, conv1_strides, name='conv1')

    pool1_strides = [1, 2, 2, 1]
    all_strides.append(pool1_strides)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=pool1_strides,
                           padding='SAME', name='max_pool1')

    activation1 = tf.nn.relu(pool1, name='activation1')

    # Convolutional layer 2.
    filter2_shape = ConvFilterShape(filter_height=3, filter_width=3,
                                    in_channels=32, out_channels=16)
    conv2_strides = [1, 2, 2, 1]
    all_strides.append(conv2_strides)
    conv2 = create_conv_layer(activation1, filter2_shape, conv2_strides, name='conv2')

    pool2_strides = [1, 3, 3, 1]
    all_strides.append(pool2_strides)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=pool2_strides,
                           padding='SAME', name='max_pool2')

    activation2 = tf.nn.relu(pool2, name='activation2')

    # Compute output size of the convolutional layers
    channels = filter2_shape.out_channels
    # Fully connected layer.
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    for e in all_strides:
        height = np.ceil(float(height) / float(e[1]))
        width = np.ceil(float(width) / float(e[2]))

    conv_out_size = int(height * width * channels)
    logging.info('Convolutional layers output {}-dimensional feature.'.format(conv_out_size))

    # Flatten the feature maps
    output = tf.reshape(activation2, [-1, conv_out_size])

    out_size = 1024
    with tf.name_scope('fc1'):
        weights = weight_variable([conv_out_size, out_size], regularization=True)
        biases = bias_variable([out_size])

        fc1 = tf.add(tf.matmul(output, weights), biases, name='fc1')

    return fc1


def main(unused_argv):
    """
    The training procedure.
    :return:
    """
    reader = DataTFReader(num_classes=NUM_CLASSES)
    train_data_pipeline = DataPipeline(reader=reader, data_pattern=FLAGS.train_data_pattern,
                                       batch_size=FLAGS.batch_size, num_threads=FLAGS.num_threads)

    # Change Me!
    tr_data_fn = flatten
    tr_data_paras = {'reshape': True, 'size': IMAGE_SIZE}

    log_reg = LogisticRegression(logdir=FLAGS.logdir)
    log_reg.fit(train_data_pipeline,
                raw_feature_size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                start_new_model=FLAGS.start_new_model, decay_steps=NUM_TRAIN_IMAGES,
                tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras)


if __name__ == '__main__':
    logging.set_verbosity(tf.logging.DEBUG)

    flags.DEFINE_string('train_data_pattern', TRAIN_TF_DATA_FILE_NAME,
                        'The Glob pattern to training data tfrecord files.')

    flags.DEFINE_integer('batch_size', 128, 'The training batch size.')

    flags.DEFINE_integer('num_threads', 2, 'The number of threads to read the tfrecord file.')

    flags.DEFINE_string('validation_data_pattern', VALIDATION_TF_DATA_FILE_NAME,
                        'The Glob pattern to validation data tfrecord files.')

    flags.DEFINE_bool('start_new_model', True, 'Whether to start a new model.')

    flags.DEFINE_string('logdir', '/tmp/log_reg', 'The log dir to log events and checkpoints.')

    app.run()
