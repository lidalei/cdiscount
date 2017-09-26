import numpy as np
import tensorflow as tf
from tensorflow import flags, logging, app
from tensorflow.contrib import slim
from pickle import load as pickle_load

from read_data import DataTFReader
from constants import NUM_TRAIN_IMAGES, NUM_CLASSES, DataPipeline, ConvFilterShape, compute_accuracy
from constants import TRAIN_TF_DATA_FILE_NAME, VALIDATION_PICKLE_DATA_FILE_NAME
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, IMAGE_SIZE

from linear_model import LogisticRegression

import inception_resnet_v2 as inception

from functools import reduce
from operator import mul


FLAGS = flags.FLAGS


def tr_log_reg_fn(input_val, **kwargs):
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
    tf.summary.histogram('model/weights', weights)
    if regularization:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)

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
    biases = tf.Variable(initial_value=initial, name=name)
    tf.summary.histogram('model/biases', biases)

    return biases


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
    # Do not forget to cast images to float type.
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


def transfer_learn_inception_resnet_v2(inputs, **kwargs):
    """
    Fine-tune the inception residual net
    :param inputs: The features batch, dimension [batch_size x height x width x channels]
    :return: The flattened just before the softmax layer.
    """
    arg_scope = inception.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        # Do not forget to cast images to float type.
        input_val = tf.cast(inputs, tf.float32)
        # num_classes is not used here, keep it small.
        # If output_stride is 8, create_aux_logits. If 16, not create_aux_logits.
        logits, end_points = inception.inception_resnet_v2(input_val, num_classes=2,
                                                           is_training=True,
                                                           create_aux_logits=False)
        tr_features = end_points['PreLogitsFlatten']

        return tr_features


def main(unused_argv):
    """
    The training procedure.
    :return:
    """
    reader = DataTFReader(num_classes=NUM_CLASSES)

    with open(FLAGS.validation_data_file, 'rb') as pickle_f:
        val_data, val_labels = pickle_load(pickle_f)

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=FLAGS.train_data_pattern,
                                       batch_size=FLAGS.batch_size, num_threads=FLAGS.num_threads)

    # Change Me!
    tr_data_fn = transfer_learn_inception_resnet_v2
    # If output_stride is 16, 1536, Conv2d_7b_1x1
    # If output_stride is 8, 3 * 3 * 1088, PreAuxlogits
    tr_data_paras = {'reshape': True, 'size': 1536}

    log_reg = LogisticRegression(logdir=FLAGS.logdir)
    log_reg.fit(train_data_pipeline,
                raw_feature_size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                start_new_model=FLAGS.start_new_model,
                tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                validation_set=(val_data, val_labels), validation_fn=compute_accuracy,
                init_learning_rate=0.00001, decay_steps=NUM_TRAIN_IMAGES * 2)


if __name__ == '__main__':
    logging.set_verbosity(tf.logging.DEBUG)

    flags.DEFINE_string('train_data_pattern', TRAIN_TF_DATA_FILE_NAME,
                        'The Glob pattern to training data tfrecord files.')

    flags.DEFINE_integer('batch_size', 32, 'The training batch size.')

    flags.DEFINE_integer('num_threads', 2, 'The number of threads to read the tfrecord file.')

    flags.DEFINE_string('validation_data_file', VALIDATION_PICKLE_DATA_FILE_NAME,
                        'The pickle file which stores the validation set.')

    flags.DEFINE_bool('start_new_model', True, 'Whether to start a new model.')

    flags.DEFINE_string('logdir', '/tmp/log_reg', 'The log dir to log events and checkpoints.')

    app.run()
