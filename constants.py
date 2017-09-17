import tensorflow as tf
from collections import namedtuple


NUM_CLASSES = 5270
CATEGORY_NAMES_FILE_NAME = '/Users/Sophie/Documents/cdiscount/category_names.csv'
BSON_DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/train_example.bson'
TRAIN_TF_DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/train.tfrecord'
VALIDATION_TF_DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/validation.tfrecord'

IMAGE_WIDTH = 180
IMAGE_HEIGHT = 180
IMAGE_CHANNELS = 3
IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

DataPipeline = namedtuple('DataPipeline', ['reader', 'data_pattern', 'batch_size', 'num_threads'])


def make_summary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

