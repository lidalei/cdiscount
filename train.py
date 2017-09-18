import tensorflow as tf
from tensorflow import flags, logging, app

from read_data import DataTFReader, get_input_data_tensors
from constants import NUM_CLASSES, DataPipeline
from constants import TRAIN_TF_DATA_FILE_NAME, VALIDATION_TF_DATA_FILE_NAME
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, IMAGE_SIZE

from linear_model import LogisticRegression

FLAGS = flags.FLAGS


def tr_data_fn(value, **kwargs):
    with tf.name_scope('reshape'):
        return tf.cast(tf.reshape(value, [-1, IMAGE_SIZE]), tf.float32)


def main(unused_argv):
    """
    The training procedure.
    :return:
    """
    reader = DataTFReader(num_classes=NUM_CLASSES)
    train_data_pipeline = DataPipeline(reader=reader, data_pattern=FLAGS.train_data_pattern,
                                       batch_size=FLAGS.batch_size, num_threads=FLAGS.num_threads)

    tr_data_paras = {'reshape': True, 'size': IMAGE_SIZE}

    log_reg = LogisticRegression(logdir=FLAGS.logdir)
    log_reg.fit(train_data_pipeline, raw_feature_size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                start_new_model=FLAGS.start_new_model, tr_data_fn=tr_data_fn,  tr_data_paras=tr_data_paras)


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
