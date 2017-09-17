"""
File is stored in bson format. It is supported by pymongo package.
From https://www.kaggle.com/inversion/processing-bson-files.

How to run the program.
'python read_data.py --category_csv_path=/home/datasets/cdiscount/category_names.csv \
--bson_data_path=/home/datasets/cdiscount/train.bson \
--train_data_pattern=/home/datasets/cdiscount/train.tfrecord \
--validation_data_pattern=/home/datasets/cdiscount/validation.tfrecord'
"""
import csv
import bson
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow import gfile, logging, flags, app
from tensorflow.python.lib.io.python_io import TFRecordWriter
from constants import NUM_CLASSES, CATEGORY_NAMES_FILE_NAME, BSON_DATA_FILE_NAME
from constants import TRAIN_TF_DATA_FILE_NAME, VALIDATION_TF_DATA_FILE_NAME

FLAGS = flags.FLAGS


class Category(object):
    def __init__(self, path):
        self.path = path
        self.mapping = dict()
        # Read the content of the csv file that defines the mapping from category id to name.
        with open(self.path, mode='rb', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            for row in reader:
                category_id = int(row['category_id'])
                del row['category_id']
                self.mapping[category_id] = row

    def get_name(self, category_id):
        """
        :param category_id: A Python int.
        :return: A Python str that represents the name of the category_id.
        """
        if category_id in self.mapping:
            return self.mapping[category_id]
        else:
            return None


class BsonReader(object):
    def __init__(self, path):
        """
        # From tensorflow/examples/how_tos/reading_data/convert_to_records.py
        :param path: The path to the bson file.
        """
        self.path = path
        self.data = bson.decode_file_iter(open(self.path, 'rb'))

    @staticmethod
    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def convert_to_tfrecord(self, category_id_mapping,
                            filenames=('train.tfrecord', 'validation.tfrecord'),
                            ratios=(0.7, 0.2)):
        """
        :param category_id_mapping: The mapping from category_id to class label (from 0 to 5269).
        :param filenames: The path to which the converted tfrecord file should be written.
        :param ratios: Splitting the whole dataset into train and validation.
            How many examples should be in the validation set.
        :return:
        """
        assert (len(filenames) == len(ratios)) and (len(ratios) == 2)
        ratio = ratios[0] / sum(ratios)
        with TFRecordWriter(filenames[0]) as tfwriter1, TFRecordWriter(filenames[1]) as tfwriter2:
            for c, d in enumerate(self.data):
                feature = {
                    '_id': self._int64_feature(d['_id']),
                    'category_id': self._int64_feature(category_id_mapping[d['category_id']])
                }

                for img in d['imgs']:
                    feature['img'] = self._bytes_feature(img['picture'])
                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    if np.random.rand(1) <= ratio:
                        tfwriter1.write(example.SerializeToString())
                    else:
                        tfwriter2.write(example.SerializeToString())

        return True


class DataTFReader(object):
    """

    """
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes

    def prepare_reader(self, filename_queue, batch_size=1024, name='examples'):
        reader = tf.TFRecordReader()
        # read_up_to return (keys, values), whose shape is ([D], [D]).
        # serialized_examples is a 1-D string Tensor.
        _, serialized_example = reader.read(filename_queue, name=name)

        feature_map = {
            '_id': tf.FixedLenFeature([], tf.int64),
            'category_id': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string)
        }

        features = tf.parse_single_example(serialized_example, features=feature_map)

        img_id = features['_id']

        img = tf.image.decode_jpeg(features['img'], channels=3)
        img.set_shape([180, 180, None])

        label = features['category_id']

        one_hot_label = tf.one_hot(label, depth=self.num_classes,
                                   on_value=1.0, off_value=0.0,
                                   dtype=tf.float32, axis=-1)

        return img_id, img, one_hot_label


def get_input_data_tensors(reader, data_pattern=None, batch_size=1024, num_threads=1, shuffle=False,
                           num_epochs=1, name_scope='input'):
    """Creates the section of the graph which reads the input data.

    Similar to the same-name function in train.py.
    Args:
        reader: A class which parses the input data.
        data_pattern: A 'glob' style path to the data files.
        batch_size: How many examples to process at a time.
        num_threads: How many I/O threads to use.
        shuffle: Boolean argument indicating whether shuffle examples.
        num_epochs: How many passed to go through the data files.
        name_scope: An identifier of this code.

    Returns:
        A tuple containing the features tensor, labels tensor.
        The exact dimensions depend on the reader being used.

    Raises:
        IOError: If no files matching the given pattern were found.
    """
    # Adapted from namesake function in inference.py.
    with tf.name_scope(name_scope):
        # Glob() can be replace with tf.train.match_filenames_once(), which is an operation.
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find input files. data_pattern='{}'".format(data_pattern))
        logging.info("Number of input files: {} within {}".format(len(files), name_scope))
        # Pass test data once. Thus, num_epochs is set as 1.
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=shuffle, capacity=32)
        example = reader.prepare_reader(filename_queue)

        # In shuffle_batch_join,
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        if shuffle:
            capacity = (num_threads + 1) * batch_size + 1024
            id_batch, image_batch, category_batch = (
                tf.train.shuffle_batch(example, batch_size,
                                       capacity, min_after_dequeue=batch_size,
                                       num_threads=num_threads,
                                       enqueue_many=False))
        else:
            capacity = num_threads * batch_size + 1024
            id_batch, image_batch, category_batch = (
                tf.train.batch(example, batch_size, num_threads=num_threads,
                               capacity=capacity,
                               allow_smaller_final_batch=True,
                               enqueue_many=False))

        return id_batch, image_batch, category_batch


def convert_bson_to_tfrecord(unused_argv):
    # Parse the mappings from category_id to category names in three levels.
    category = Category(FLAGS.category_csv_path)
    print('{}: {}'.format(1000012776, category.get_name(1000012776)))
    category_ids = category.mapping.keys()
    print('category_id, max {}, min {}'.format(max(category_ids), min(category_ids)))
    category_id_mapping = dict(zip(sorted(category_ids), range(len(category_ids))))

    # Convert bson file to tfrecord files
    bson_reader = BsonReader(FLAGS.bson_data_path)
    bson_reader.convert_to_tfrecord(category_id_mapping,
                                    filenames=(FLAGS.train_data_pattern,
                                               FLAGS.validation_data_pattern),
                                    ratios=(0.7, 0.2))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)

    flags.DEFINE_string('category_csv_path', CATEGORY_NAMES_FILE_NAME,
                        'The path to the category csv file.')

    flags.DEFINE_string('bson_data_path', BSON_DATA_FILE_NAME,
                        'The path to the bson data file.')

    flags.DEFINE_string('train_data_pattern', TRAIN_TF_DATA_FILE_NAME,
                        'The Glob pattern to training data tfrecord files.')

    flags.DEFINE_string('validation_data_pattern', VALIDATION_TF_DATA_FILE_NAME,
                        'The Glob pattern to validation data tfrecord files.')

    app.run(main=convert_bson_to_tfrecord, argv=None)
