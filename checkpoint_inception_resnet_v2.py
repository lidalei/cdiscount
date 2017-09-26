"""
To get the appropriate pre-trained saver.
"""


import tensorflow as tf
from tensorflow.contrib import slim

import inception_resnet_v2 as inception
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS


def main():
    """
    Fine-tune the inception residual net
    :return: The flattened just before the softmax layer.
    """
    with tf.Graph().as_default() as g:
        arg_scope = inception.inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            # num_classes is not used here, keep it small.
            # If output_stride is 8, create_aux_logits. If 16, not create_aux_logits.
            logits, end_points = inception.inception_resnet_v2(inputs,
                                                               is_training=True,
                                                               create_aux_logits=False)
        saver = tf.train.Saver()

    with tf.Session(graph=g) as sess:
        saver.restore(sess, 'inception_resnet_v2_model/inception_resnet_v2_2016_08_30.ckpt')

        saver.save(sess, 'inception_resnet_v2_model/inception_resnet_v2.ckpt')

if __name__ == '__main__':
    main()
