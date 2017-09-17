import tensorflow as tf


NUM_CLASSES = 5270
CATEGORY_NAMES_FILE_NAME = '/Users/Sophie/Documents/cdiscount/category_names.csv'
BSON_DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/train_example.bson'
TRAIN_TF_DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/train.tfrecord'
VALIDATION_TF_DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/validation.tfrecord'


def make_summary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

