"""
File is stored in bson format. It is supported by pymongo package.
From https://www.kaggle.com/inversion/processing-bson-files.
"""
import csv
import bson
import matplotlib.pylab as plt
from skimage.data import imread
import io
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import gfile, logging


DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/train_example.bson'
CATEGORY_NAMES_FILE_NAME = '/Users/Sophie/Documents/cdiscount/category_names.csv'


class Category(object):
    def __init__(self):
        self.mapping = dict()
        # Read the content of the csv file that defines the mapping from category id to name.
        with open(CATEGORY_NAMES_FILE_NAME, newline='') as csv_file:
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

    def convert_to_tfrecord(self, filename, category_id_mapping):
        """
        :param filename: The path to which the converted tfrecord file should be written.
        :return:
        """
        with tf.python_io.TFRecordWriter(filename) as tfwriter:
            for c, d in enumerate(self.data):
                feature = {
                    '_id': self._int64_feature(d['_id']),
                    'category_id': self._int64_feature(category_id_mapping[d['category_id']]),
                    'imgs': self._bytes_feature([img['picture'] for img in d['imgs']])
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                tfwriter.write(example.SerializeToString())


class DataTFReader(object):
    """

    """
    def __init__(self, num_classes=5000):
        self.num_classes = num_classes

    def prepare_reader(self, filename_queue, batch_size=1024, name='examples'):
        reader = tf.TFRecordReader()
        # read_up_to return (keys, values), whose shape is ([D], [D]).
        # serialized_examples is a 1-D string Tensor.
        _, serialized_examples = reader.read_up_to(filename_queue, batch_size, name=name)

        feature_map = {
            '_id': tf.FixedLenFeature([], tf.int64),
            'category_id': tf.FixedLenFeature([], tf.int64),
            'imgs': tf.VarLenFeature(tf.string)
        }

        features = tf.parse_example(serialized_examples, features=feature_map)

        ids = features['_id']
        imgs = features['imgs']
        # TODO
        # imgs = tf.image.decode_jpeg(features['imgs'])

        labels = features['category_id']

        one_hot_labels = tf.one_hot(labels, depth=self.num_classes, on_value=1, off_value=0, axis=-1)

        return ids, imgs, one_hot_labels


def get_input_data_tensors(reader, data_pattern=None, batch_size=2048, num_threads=1, shuffle=False,
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
        examples_and_labels = reader.prepare_reader(filename_queue)

        # In shuffle_batch_join,
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        if shuffle:
            capacity = (num_threads + 1) * batch_size + 2048
            id_batch, features_batch, labels_batch = (
                tf.train.shuffle_batch(examples_and_labels, batch_size,
                                       capacity, min_after_dequeue=batch_size,
                                       num_threads=num_threads,
                                       enqueue_many=True))
        else:
            capacity = num_threads * batch_size + 2048
            id_batch, features_batch, labels_batch = (
                tf.train.batch(examples_and_labels, batch_size, num_threads=num_threads,
                               capacity=capacity,
                               allow_smaller_final_batch=True,
                               enqueue_many=True))

        return id_batch, features_batch, labels_batch


if __name__ == '__main__':
    # Parse the mappings from category_id to category names in three levels.
    category = Category()
    print('{}: {}'.format(1000012776, category.get_name(1000012776)))
    category_ids = category.mapping.keys()
    print('category_id, max {}, min {}'.format(max(category_ids), min(category_ids)))
    category_id_mapping = dict(zip(sorted(category_ids), range(len(category_ids))))

    # Convert bson file to tfrecord files
    bson_reader = BsonReader(DATA_FILE_NAME)
    bson_reader.convert_to_tfrecord('/tmp/bson.tfrecord', category_id_mapping)

    g = tf.Graph()

    with g.as_default() as g:
        tf_reader = DataTFReader(num_classes=5000)
        id_batch, features_batch, labels_batch = get_input_data_tensors(
            tf_reader, data_pattern='/tmp/*.tfrecord', batch_size=10)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                           name='init_glo_loc_var')

    with tf.Session(graph=g) as sess:
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                id_batch_val, features_batch_val, labels_batch_val = sess.run([id_batch, features_batch, labels_batch])
                print(id_batch_val, features_batch_val, labels_batch_val)
                coord.request_stop()
        except tf.errors.OutOfRangeError:
            logging.info('Finished normal equation terms computation -- one epoch done.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
