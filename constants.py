import numpy as np
import tensorflow as tf
from collections import namedtuple
from os.path import join as path_join


NUM_TRAIN_IMAGES = 12371293
NUM_CLASSES = 5270

DATA_SET_FOLDER = '/Users/Sophie/Documents/cdiscount/'
CATEGORY_NAMES_FILE_NAME = path_join(DATA_SET_FOLDER, 'category_names.csv')
BSON_DATA_FILE_NAME = path_join(DATA_SET_FOLDER, 'train_example.bson')
TRAIN_TF_DATA_FILE_NAME = path_join(DATA_SET_FOLDER, 'train.tfrecord')
VALIDATION_TF_DATA_FILE_NAME = path_join(DATA_SET_FOLDER, 'validation.tfrecord')
TEST_TF_DATA_FILE_NAME = path_join(DATA_SET_FOLDER, 'test.tfrecord')

IMAGE_WIDTH = 180
IMAGE_HEIGHT = 180
IMAGE_CHANNELS = 3
IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

DataPipeline = namedtuple('DataPipeline', ['reader', 'data_pattern', 'batch_size', 'num_threads'])
ConvFilterShape = namedtuple('ConvFilterShape', ['filter_height', 'filter_width', 'in_channels', 'out_channels'])


def make_summary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary


def compute_accuracy(labels=None, predictions=None):
    """
    Compute accuracy for a batch of labels and predictions.
    Each element is treated as an example.
    :param labels: The true labels.
    :param predictions: The predicted labels.
    :return: The accuracy.
    """
    return np.sum(np.equal(labels, predictions)) / np.size(labels)

