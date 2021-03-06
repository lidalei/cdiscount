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
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import gfile, logging, flags, app
from tensorflow.python.lib.io.python_io import TFRecordWriter
from constants import NUM_CLASSES, CATEGORY_NAMES_FILE_NAME, BSON_DATA_FILE_NAME, DataPipeline
from constants import TRAIN_TF_DATA_FILE_NAME, VALIDATION_TF_DATA_FILE_NAME
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

FLAGS = flags.FLAGS


class Category(object):
    def __init__(self, path):
        self.path = path
        # Mapping rom category_id to three-level names
        self.mapping = dict()
        # Read the content of the csv file that defines the mapping from category id to name.
        with open(self.path, mode='r') as csv_file:
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
        self.data = bson.decode_file_iter(open(self.path, mode='rb'))

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
        if tf.gfile.Exists(filenames[0]) or tf.gfile.Exists(filenames[1]):
            raise FileExistsError('File exists. Continuing will overwrite it. Abort!')

        ratio = ratios[0] / sum(ratios)
        with TFRecordWriter(filenames[0]) as tfwriter1, TFRecordWriter(filenames[1]) as tfwriter2:
            for c, d in enumerate(self.data):
                feature = {
                    '_id': self._int64_feature(d['_id'])
                }

                # Test dataset.
                if 'category_id' not in d:
                    feature['category_id'] = self._int64_feature(-1)
                else:
                    feature['category_id'] = self._int64_feature(category_id_mapping[d['category_id']])

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

    def prepare_reader(self, filename_queue, batch_size=1024, onehot_label=False,
                       decode_image=True, name='examples'):
        reader = tf.TFRecordReader()
        feature_map = {
            '_id': tf.FixedLenFeature([], tf.int64),
            'category_id': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string)
        }
        # Read a single example.
        # _, serialized_example = reader.read(filename_queue, name=name)
        # features = tf.parse_single_example(serialized_example, features=feature_map)
        # img_id = features['_id']
        # img = tf.image.decode_jpeg(features['img'], channels=IMAGE_CHANNELS)
        # img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, None])
        # label = features['category_id']

        # read_up_to return (keys, values), whose shape is ([D], [D]).
        # serialized_examples is a 1-D string Tensor.
        _, serialized_examples = reader.read_up_to(filename_queue, batch_size, name=name)
        features = tf.parse_example(serialized_examples, features=feature_map)
        img_ids = features['_id']

        if decode_image:
            # Users must provide dtype if it is different from the data type of elems.
            imgs = tf.map_fn(lambda img: tf.image.decode_jpeg(img, channels=IMAGE_CHANNELS), features['img'],
                             dtype=tf.uint8)
            imgs.set_shape([None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        else:
            imgs = tf.ones([tf.shape(serialized_examples)[0]], dtype=tf.uint8)

        labels = features['category_id']

        if onehot_label:
            one_hot_labels = tf.one_hot(labels, depth=self.num_classes,
                                        on_value=1, off_value=0,
                                        dtype=tf.int32, axis=-1)

            return img_ids, imgs, one_hot_labels
        else:
            return img_ids, imgs, labels


def get_input_data_tensors(data_pipeline, onehot_label=False, shuffle=False, num_epochs=1,
                           fixed_batch_size=False, decode_image=True, name_scope='input'):
    reader = data_pipeline.reader
    data_pattern = data_pipeline.data_pattern
    batch_size = data_pipeline.batch_size
    num_threads = data_pipeline.num_threads

    return _get_input_data_tensors(reader, data_pattern=data_pattern, batch_size=batch_size,
                                   num_threads=num_threads, onehot_label=onehot_label,
                                   shuffle=shuffle, num_epochs=num_epochs,
                                   fixed_batch_size=fixed_batch_size,
                                   decode_image=decode_image, name_scope=name_scope)


def _get_input_data_tensors(reader, data_pattern=None, batch_size=1024, num_threads=1,
                            onehot_label=False, shuffle=False, num_epochs=1,
                            fixed_batch_size=False, decode_image=True, name_scope='input'):
    """Creates the section of the graph which reads the input data.

    Similar to the same-name function in train.py.
    Args:
        reader: A class which parses the input data.
        data_pattern: A 'glob' style path to the data files.
        batch_size: How many examples to process at a time.
        num_threads: How many I/O threads to use.
        onehot_label: Whether return onehot encoded label.
        shuffle: Boolean argument indicating whether shuffle examples.
        num_epochs: How many passed to go through the data files.
        fixed_batch_size: If return fixed batch size.
        decode_image: Whether to decode image.
        name_scope: An identifier of this code.

    Returns:
        A tuple containing the features tensor, labels tensor.
        The exact dimensions depend on the reader being used.

    Raises:
        IOError: If no files matching the given pattern were found.
    """
    if not isinstance(decode_image, bool):
        raise ValueError('decode_image {} can either be True or False.'.format(decode_image))

    # Glob() can be replace with tf.train.match_filenames_once(), which is an operation.
    files = gfile.Glob(data_pattern)
    if not files:
        raise IOError("Unable to find input files. data_pattern='{}'".format(data_pattern))
    logging.info("Number of input files: {} within {}".format(len(files), name_scope))

    with tf.name_scope(name_scope), tf.device('/cpu:0'):
        # Pass test data num_epochs.
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs,
                                                        shuffle=shuffle, capacity=8)
        examples = reader.prepare_reader(filename_queue,
                                         onehot_label=onehot_label,
                                         decode_image=decode_image)

        # In shuffle_batch_join,
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        # allow_smaller_final_batch False to ensure there must be batch_size examples returned.
        if shuffle:
            capacity = (num_threads + 2) * batch_size
            id_batch, image_batch, category_batch = (
                tf.train.shuffle_batch(examples, batch_size,
                                       capacity, min_after_dequeue=batch_size,
                                       num_threads=num_threads,
                                       allow_smaller_final_batch=not fixed_batch_size,
                                       enqueue_many=True))
        else:
            # allow_smaller_final_batch True to ensure all test examples are handled.
            capacity = (num_threads + 2) * batch_size
            id_batch, image_batch, category_batch = (
                tf.train.batch(examples, batch_size, num_threads=num_threads,
                               capacity=capacity,
                               allow_smaller_final_batch=not fixed_batch_size,
                               enqueue_many=True))

        return id_batch, image_batch, category_batch


def convert_bson_to_tfrecord(_):
    # Parse the mappings from category_id to category names in three levels.
    category = Category(FLAGS.category_csv_path)
    print('{}: {}'.format(1000012776, category.get_name(1000012776)))
    category_ids = category.mapping.keys()
    print('category_id, max {}, min {}'.format(max(category_ids), min(category_ids)))
    category_id_mapping = dict(zip(sorted(category_ids), range(len(category_ids))))

    ratios = [float(e) for e in FLAGS.train_val_ratios.split(',')]

    # Convert bson file to tfrecord files
    bson_reader = BsonReader(FLAGS.bson_data_path)
    bson_reader.convert_to_tfrecord(category_id_mapping,
                                    filenames=(FLAGS.train_data_pattern,
                                               FLAGS.validation_data_pattern),
                                    ratios=ratios)


def main(_):
    """
    The training procedure.
    :return:
    """
    reader = DataTFReader(num_classes=NUM_CLASSES)
    data_pipeline = DataPipeline(reader, FLAGS.train_data_pattern, 1024, 4)

    g = tf.Graph()
    with g.as_default() as g:
        global_step = tf.Variable(0, trainable=False, name='global_step')
        num_images = tf.Variable(0, dtype=tf.int32, name='num_images')
        sum_labels_onehot = tf.Variable(tf.zeros([NUM_CLASSES], dtype=tf.int32), name='sum_labels')
        sum_pixels = tf.Variable(tf.zeros([IMAGE_CHANNELS], dtype=tf.int64), name='sum_pixels')

        global_step_inc_op = global_step.assign_add(1)
        # decoding images can be avoided
        id_batch, image_batch, label_batch = get_input_data_tensors(
            data_pipeline, onehot_label=True, shuffle=False, decode_image=True)

        num_images_op = num_images.assign_add(tf.shape(id_batch)[0])

        sum_labels_onehot_op = sum_labels_onehot.assign_add(
            tf.reduce_sum(tf.cast(label_batch, tf.int32), axis=0))

        sum_pixels_op = sum_pixels.assign_add(
            tf.reduce_sum(tf.cast(image_batch, tf.int64), axis=[0, 1, 2]))

        with tf.control_dependencies(
                [global_step_inc_op, num_images_op, sum_labels_onehot_op, sum_pixels_op]):
            accum_non_op = tf.no_op()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                           name='init_glo_loc_var')

        # summary_op = tf.summary.merge_all()

    with tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)

        summary_writer = tf.summary.FileWriter('/tmp/statistics', graph=sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _, global_step_val = sess.run([accum_non_op, global_step])
                if global_step_val % 10 == 0:
                    print('Step: {}'.format(global_step_val))
                # id_batch_val, image_batch_val, label_batch_val, summary = sess.run(
                #     [id_batch, image_batch, label_batch, summary_op])
                # logging.debug('id: {}, image: {}, label: {}'.format(
                #     id_batch_val, image_batch_val[0], label_batch_val))
                # summary_writer.add_summary(summary)
                # coord.request_stop()
        except tf.errors.OutOfRangeError:
            logging.info('One epoch done.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        num_images_val = sess.run(num_images)
        sum_labels_onehot_val = sess.run(sum_labels_onehot)
        sum_pixels_val = sess.run(sum_pixels)

        print('num images: {}'.format(num_images_val))
        # print('sum labels onehot: {}'.format(','.join(sum_labels_onehot_val.astype(np.str))))
        print('mean pixels: {}'.format(
            sum_pixels_val / (num_images_val * IMAGE_HEIGHT * IMAGE_WIDTH)))

        with open('sum_labels_one_hot.pickle', mode='wb') as f:
            pickle.dump(sum_labels_onehot_val, f)

        with open('mean_pixels.pickle', 'wb') as f:
            pickle.dump(sum_pixels_val / (num_images_val * IMAGE_HEIGHT * IMAGE_WIDTH), f)

        summary_writer.close()

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

    flags.DEFINE_string('train_val_ratios', '0.7,0.2',
                        'The percentage of train and validation partition.')

    app.run(main)
