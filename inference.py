"""
This file contains BootstrapInference class. This class is used to make predictions and
 averaging predictions from many models to form the final prediction.
Note: The function format_lines is copied from inference.py in the same parent folder.
    It is just for decoupling file dependency.
"""
import tensorflow as tf
from tensorflow import logging, gfile, flags, app

import numpy as np
from read_data import get_input_data_tensors, DataPipeline

import time

from read_data import DataTFReader, Category
from constants import NUM_CLASSES, TEST_TF_DATA_FILE_NAME, CATEGORY_NAMES_FILE_NAME

from pyspark.conf import SparkConf
from pyspark.context import SparkContext


FLAGS = flags.FLAGS


class BootstrapInference(object):
    def __init__(self, train_model_dirs_list, sorted_category_ids):
        """
        :param train_model_dirs_list: The list of paths that contain the saved models.
        :param sorted_category_ids: The sorted category_ids,
            which is used to convert label to category.
        """
        # Bagging, load several trained models and average the predictions.
        self.train_model_dirs_list = train_model_dirs_list
        self.sorted_category_ids = sorted_category_ids

        self.sess_list = []
        self.img_input_batch_list = []
        self.pred_prob_list = []
        self.phase_train_pl_list = []

        # Use PySpark
        self.conf = SparkConf()
        self.conf.setMaster("local[*]").setAppName("Bootstrap Inference")
        self.sc = SparkContext(conf=self.conf)

        for train_model_dir in train_model_dirs_list:
            # Load pre-trained graph and corresponding variables.
            g = tf.Graph()
            with g.as_default():
                latest_checkpoint = tf.train.latest_checkpoint(train_model_dir)
                if latest_checkpoint is None:
                    raise Exception("unable to find a checkpoint at location: {}".format(train_model_dir))
                else:
                    meta_graph_location = '{}{}'.format(latest_checkpoint, ".meta")
                    logging.info("loading meta-graph: {}".format(meta_graph_location))
                pre_trained_saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)

                # Create a session to restore model parameters.
                sess = tf.Session(graph=g)
                logging.info("restoring variables from {}".format(latest_checkpoint))
                pre_trained_saver.restore(sess, latest_checkpoint)
                # Get collections to be used in making predictions for test data.
                img_input_batch = tf.get_collection('raw_features_batch')[0]
                pred_prob = tf.get_collection('pred_prob')[0]
                # phase_train_pl collection might be empty.
                phase_train_pl = tf.get_collection('phase_train_pl')

                # Append session and input and predictions.
                self.sess_list.append(sess)
                self.img_input_batch_list.append(img_input_batch)
                self.pred_prob_list.append(pred_prob)
                if len(phase_train_pl) >= 1:
                    self.phase_train_pl_list.append({phase_train_pl[0]: False})
                else:
                    self.phase_train_pl_list.append({})

    def __del__(self):
        for sess in self.sess_list:
            sess.close()

    @staticmethod
    def majority_voting(array):
        unique_vals, counts = np.unique(array, return_counts=True)
        return unique_vals[np.argmax(counts)]

    def transform(self, test_data_pipeline, out_file_location):
        test_graph = tf.Graph()
        with test_graph.as_default():
            id_batch, img_batch, label_batch = (
                get_input_data_tensors(test_data_pipeline, shuffle=False,
                                       num_epochs=1, name_scope='test_input'))

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Run test graph to get video batch and feed video batch to pre_trained_graph to get predictions.
        test_sess = tf.Session(graph=test_graph)
        with gfile.Open(out_file_location, "w+") as out_file:
            test_sess.run(init_op)

            # Be cautious to not be blocked by queue.
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=test_sess, coord=coord)

            out_file.write("_id,category_id\n")
            processing_count, num_examples_processed = 0, 0
            ids_val, pred_probs_val = [], []

            try:
                while not coord.should_stop():
                    # Run training steps or whatever.
                    start_time = time.time()
                    id_batch_val, img_batch_val = test_sess.run([id_batch, img_batch])
                    logging.debug('id_batch_val: {}\nimg_batch_val: {}'.format(
                        id_batch_val, img_batch_val))
                    # In case fixed batch size is required.
                    # Some batches have less than batch size examples.
                    batch_size = test_data_pipeline.batch_size
                    id_batch_val_length = len(id_batch_val)
                    if id_batch_val_length < batch_size:
                        indices = np.arange(batch_size) % id_batch_val_length
                        id_batch_val = id_batch_val[indices]
                        img_batch_val = img_batch_val[indices]

                    batch_pred_prob_list = []
                    for sess, img_input_batch, pred_prob, phase_train_pl in zip(
                            self.sess_list, self.img_input_batch_list,
                            self.pred_prob_list, self.phase_train_pl_list):
                        # logging.info('Feature shape is {}.'.format(feature_shape))

                        batch_pred_prob = sess.run(pred_prob, feed_dict=dict(
                            {img_input_batch: img_batch_val, **phase_train_pl}
                        ))
                        batch_pred_prob_list.append(batch_pred_prob)

                    batch_pred_mean_prob = np.mean(np.stack(batch_pred_prob_list, axis=0), axis=0)
                    batch_pred = np.argmax(batch_pred_mean_prob, axis=-1)
                    # Write batch predictions to files.
                    # Be cautious! One product can have multiple images
                    ids_val.extend(id_batch_val)
                    pred_probs_val.extend(batch_pred)

                    now = time.time()
                    processing_count += 1
                    num_examples_processed += id_batch_val.shape[0]
                    if processing_count % 10 == 0:
                        print('Step {}, elapsed {} s, processed {} examples in total'.format(
                            processing_count, now - start_time, num_examples_processed))

            except tf.errors.OutOfRangeError:
                logging.info('Done predictions.')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

            id_category_rdd = self.sc.parallelize(zip(ids_val, pred_probs_val))

            # Append and extend return None!!! Instead, use + which returns a new list.
            id_ped_labels = id_category_rdd.aggregateByKey(
                list(),
                lambda x, y: x + [y],
                lambda x, y: x + y).mapValues(self.majority_voting).sortByKey().collect()

            # Don't put write into foreach loop, for it does  not work.
            for id_pred_label in id_ped_labels:
                out_file.write('{},{}\n'.format(id_pred_label[0], self.sorted_category_ids[id_pred_label[1]]))

            out_file.flush()

            test_sess.close()
            out_file.close()
            print('All predictions were written to {}'.format(out_file_location))


def main(unsed_argv):
    logging.set_verbosity(logging.INFO)
    # Where training checkpoints are stored.
    train_model_dirs = FLAGS.train_model_dirs

    reader = DataTFReader(num_classes=NUM_CLASSES)
    # Get test data.
    test_data_pipeline = DataPipeline(reader=reader, data_pattern=FLAGS.test_data_pattern,
                                      batch_size=FLAGS.batch_size, num_threads=FLAGS.num_threads)

    train_model_dirs_list = [e.strip() for e in train_model_dirs.split(',')]

    category = Category(FLAGS.category_csv_path)
    print('{}: {}'.format(1000012776, category.get_name(1000012776)))
    sorted_category_ids = np.array(sorted(category.mapping.keys()))
    print('category_id, max {}, min {}'.format(sorted_category_ids[-1], sorted_category_ids[0]))

    # Make inference.
    inference = BootstrapInference(train_model_dirs_list, sorted_category_ids)
    inference.transform(test_data_pipeline, FLAGS.output_file)


if __name__ == '__main__':
    flags.DEFINE_string('category_csv_path', CATEGORY_NAMES_FILE_NAME,
                        'The path to the category csv file.')

    flags.DEFINE_string('test_data_pattern',
                        TEST_TF_DATA_FILE_NAME,
                        'Test data pattern, to be specified when making predictions.')

    flags.DEFINE_integer('batch_size', 512, 'Size of batch processing.')
    flags.DEFINE_integer('num_threads', 2, 'Number of readers to form a batch.')

    # Separated by , (csv separator), e.g., log_reg,conv_net. Used in bagging.
    flags.DEFINE_string('train_model_dirs', '/tmp/inception_v4/log_reg/',
                        'The directories where to load trained logistic regression models.')

    flags.DEFINE_string('output_file', '/tmp/predictions_{}.csv'.format(int(time.time())),
                        'The file to save the predictions to.')

    app.run()
