import tensorflow as tf
from tensorflow import flags, logging, app

from read_data import DataTFReader, get_input_data_tensors
from constants import NUM_CLASSES
from constants import TRAIN_TF_DATA_FILE_NAME, VALIDATION_TF_DATA_FILE_NAME

FLAGS = flags.FLAGS


def main():
    """
    The training procedure.
    :return:
    """
    g = tf.Graph()
    with g.as_default() as g:
        tf_reader = DataTFReader(num_classes=NUM_CLASSES)
        id_batch, image_batch, label_batch = get_input_data_tensors(
            tf_reader, data_pattern=FLAGS.train_data_pattern, batch_size=100)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                           name='init_glo_loc_var')

        summary_op = tf.summary.merge_all()

    with tf.Session(graph=g) as sess:
        sess.run(init_op)

        summary_writer = tf.summary.FileWriter('/tmp', graph=sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                id_batch_val, image_batch_val, label_batch_val, summary = sess.run(
                    [id_batch, image_batch, label_batch, summary_op])
                logging.debug('id: {}, image: {}, label: {}'.format(
                    id_batch_val, image_batch_val[0], label_batch_val))
                summary_writer.add_summary(summary)
                coord.request_stop()
        except tf.errors.OutOfRangeError:
            logging.info('Finished normal equation terms computation -- one epoch done.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        summary_writer.close()

if __name__ == '__main__':
    logging.set_verbosity(tf.logging.DEBUG)

    flags.DEFINE_string('train_data_pattern', TRAIN_TF_DATA_FILE_NAME,
                        'The Glob pattern to training data tfrecord files.')

    flags.DEFINE_string('validation_data_pattern', VALIDATION_TF_DATA_FILE_NAME,
                        'The Glob pattern to validation data tfrecord files.')
    app.run()
