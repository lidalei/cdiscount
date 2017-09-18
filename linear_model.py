import numpy as np
import tensorflow as tf
from tensorflow import logging

# from constants import make_summary
from read_data import get_input_data_tensors


class LogisticRegression(object):
    def __init__(self, logdir='/tmp/log_reg', max_train_steps=1000000):
        """
        Args:
             logdir: The dir where intermediate results and model checkpoints should be written.
        """
        self.logdir = logdir
        self.max_train_steps = max_train_steps

        # Member variables used to construct graph.
        self.train_data_pipeline = None
        self.raw_feature_size = None
        self.feature_size = None
        self.num_classes = None
        self.batch_size = None
        self.tr_data_fn = None
        self.tr_data_paras = dict()
        self.init_learning_rate = 0.001
        self.decay_steps = 40000
        self.decay_rate = 0.95
        self.epochs = None
        self.l1_reg_rate = None
        self.l2_reg_rate = None
        self.pos_weights = None

        self.graph = None
        # Member variables associated with graph.
        self.saver = None
        self.global_step = None
        self.init_op = None
        self.train_op = None
        self.summary_op = None
        self.raw_features_batch = None
        self.labels_batch = None
        self.loss = None
        self.pred_prob = None

    def _build_graph(self):
        """
        Build graph.

        Returns:
            A saver object. It can be used in constructing a Supervisor object.
        Note:
            To avoid contaminating default graph.
            This function must be wrapped into a with tf.Graph().as_default() as graph contextmanager.
        """
        # Build logistic regression graph and optimize it.
        # Set seed to keep whole data sampling consistency, though impossible due to system variation.
        # seed = np.random.randint(2 ** 28)
        # tf.set_random_seed(seed)

        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')

        id_batch, raw_features_batch, labels_batch = (
            get_input_data_tensors(self.train_data_pipeline, shuffle=True, num_epochs=self.epochs,
                                   name_scope='input'))

        # Define num_classes logistic regression models parameters.
        weights = tf.Variable(initial_value=tf.truncated_normal(
            [self.feature_size, self.num_classes], stddev=1.0 / np.sqrt(self.feature_size)),
            dtype=tf.float32, name='weights')

        # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
        tf.summary.histogram('model/weights', weights)

        biases = tf.Variable(initial_value=tf.zeros([self.num_classes]), name='biases')

        tf.summary.histogram('model/biases', biases)

        if self.tr_data_fn is None:
            transformed_features_batch = tf.identity(raw_features_batch)
        else:
            transformed_features_batch = self.tr_data_fn(raw_features_batch, **self.tr_data_paras)

        logits = tf.add(tf.matmul(transformed_features_batch, weights), biases, name='logits')

        pred_prob = tf.nn.softmax(logits, name='pred_probability')

        with tf.name_scope('train'):
            loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_batch, logits=logits, name='x_entropy_per_example')

            loss = tf.reduce_mean(loss_per_example, name='x_entropy')

            tf.summary.scalar('loss/xentropy', loss)

            # Add regularization.
            reg_losses = []
            # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
            to_regularize = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.l1_reg_rate and self.l1_reg_rate != 0:
                l1_reg_losses = [tf.reduce_sum(tf.abs(w)) for w in to_regularize]
                l1_reg_loss = tf.add_n(l1_reg_losses, name='l1_reg_loss')
                tf.summary.scalar('loss/l1_reg_loss', l1_reg_loss)
                reg_losses.append(tf.multiply(self.l1_reg_rate, l1_reg_loss))

            if self.l2_reg_rate and self.l2_reg_rate != 0:
                l2_reg_losses = [tf.reduce_sum(tf.square(w)) for w in to_regularize]
                l2_reg_loss = tf.add_n(l2_reg_losses, name='l2_loss')
                tf.summary.scalar('loss/l2_reg_loss', l2_reg_loss)
                reg_losses.append(tf.multiply(self.l2_reg_rate, l2_reg_loss))

            if len(reg_losses) > 0:
                reg_loss = tf.add_n(reg_losses, name='reg_loss')
            else:
                reg_loss = tf.constant(0.0, name='zero_reg_loss')

            final_loss = tf.add(loss, reg_loss, name='final_loss')

        with tf.name_scope('optimization'):
            # Decayed learning rate.
            rough_num_examples_processed = tf.multiply(global_step, self.batch_size)
            adap_learning_rate = tf.train.exponential_decay(self.init_learning_rate, rough_num_examples_processed,
                                                            self.decay_steps, self.decay_rate, staircase=True,
                                                            name='adap_learning_rate')
            tf.summary.scalar('learning_rate', adap_learning_rate)
            # GradientDescentOptimizer
            optimizer = tf.train.GradientDescentOptimizer(adap_learning_rate)
            # MomentumOptimizer
            # optimizer = tf.train.MomentumOptimizer(adap_learning_rate, 0.9, use_nesterov=True)
            # RMSPropOptimizer
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.init_learning_rate)
            train_op = optimizer.minimize(final_loss, global_step=global_step)

        summary_op = tf.summary.merge_all()
        # summary_op = tf.constant(1.0)

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Used for restoring training checkpoints.
        tf.add_to_collection('global_step', global_step)
        tf.add_to_collection('init_op', init_op)
        tf.add_to_collection('train_op', train_op)
        tf.add_to_collection('summary_op', summary_op)
        # Add to collection. In inference, get collection and feed it with test data.
        tf.add_to_collection('raw_features_batch', raw_features_batch)
        tf.add_to_collection('labels_batch', labels_batch)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('predictions', pred_prob)

        # To save global variables and savable objects, i.e., var_list is None.
        # Using rbf transform will also save centers and scaling factors.
        saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=0.15)

        return saver

    def _restore_graph(self):
        """
        Restore graph def.
        Returns:
             A saver previously created when building this graph.
        Note:
            To avoid contaminating default graph.
            This function must be wrapped into a with tf.Graph().as_default() as graph contextmanager.
        """
        # Load pre-trained graph.
        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)
        if latest_checkpoint is None:
            raise Exception("unable to find a checkpoint at location: {}".format(self.logdir))
        else:
            meta_graph_location = '{}{}'.format(latest_checkpoint, ".meta")
            logging.info("loading meta-graph: {}".format(meta_graph_location))
        # Recreates a Graph saved in a MetaGraphDef proto, docs.
        pre_trained_saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)

        return pre_trained_saver

    def _check_graph_initialized(self):
            """
            To check if all graph operations and the graph itself are initialized successfully.

            Return:
                True if graph and all graph ops are not None, otherwise False.
            """
            graph_ops = [self.saver, self.global_step, self.init_op, self.train_op, self.summary_op,
                         self.raw_features_batch, self.labels_batch, self.loss, self.pred_prob]

            return (self.graph is not None) and (graph_ops.count(None) == 0)

    def fit(self, train_data_pipeline, raw_feature_size, start_new_model=False,
            tr_data_fn=None, tr_data_paras=None,
            init_learning_rate=0.001, decay_steps=40000, decay_rate=0.95, epochs=None,
            l1_reg_rate=None, l2_reg_rate=None, pos_weights=None):
        """
        Logistic regression fit function.
        Args:
            train_data_pipeline: A namedtuple consisting of reader, data_pattern, batch_size and num_readers.
            raw_feature_size: The dimensionality of features.
            start_new_model: If True, start a new model instead of restoring from existing checkpoints.
            tr_data_fn: a function that transforms input data.
            tr_data_paras: Other parameters should be passed to tr_data_fn. A dictionary.
            init_learning_rate: Decayed gradient descent parameter.
            decay_steps: Decayed gradient descent parameter.
            decay_rate: Decayed gradient descent parameter.
            epochs: Maximal epochs to use.
            l1_reg_rate: None, not impose l1 regularization.
            l2_reg_rate: l2 regularization rate.
            pos_weights: For imbalanced binary classes. Here, num_pos << num_neg, the weights should be > 1.0.
                If None, treated as 1.0 for all binary classifiers.
        Returns: None.
        """
        reader = train_data_pipeline.reader
        batch_size = train_data_pipeline.batch_size
        num_classes = reader.num_classes
        logging.info('Logistic regression uses {}-dimensional features.'.format(raw_feature_size))

        self.train_data_pipeline = train_data_pipeline
        self.raw_feature_size = raw_feature_size
        self.feature_size = self.raw_feature_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.tr_data_fn = tr_data_fn
        self.tr_data_paras = tr_data_paras
        self.init_learning_rate = init_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.l1_reg_rate = l1_reg_rate
        self.l2_reg_rate = l2_reg_rate
        self.pos_weights = pos_weights

        # Check extra data transform function arguments.
        # If transform changes the features size, change it.
        if self.tr_data_fn is not None:
            if self.tr_data_paras is None:
                self.tr_data_paras = dict()
            else:
                if ('reshape' in self.tr_data_paras) and (self.tr_data_paras['reshape'] is True):
                    self.feature_size = self.tr_data_paras['size']
                    logging.warn('Data transform changes the features size to {}.'.format(
                        self.feature_size))

            logging.debug('Data transform arguments are {}.'.format(self.tr_data_paras))
        else:
            self.tr_data_paras = dict()

        start_new_model = start_new_model or (not tf.gfile.Exists(self.logdir))

        # This is NECESSARY to avoid contaminating default graph.
        # Alternatively, we can define a member graph variable. When building a new graph or
        # restoring a graph, wrap the code into a similar contextmanager.
        self.graph = tf.Graph()
        with self.graph.as_default():
            if start_new_model:
                logging.info('Starting a new model...')
                # Start new model, delete existing checkpoints.
                if tf.gfile.Exists(self.logdir):
                    try:
                        tf.gfile.DeleteRecursively(self.logdir)
                    except tf.errors.OpError:
                        logging.error('Failed to delete dir {}.'.format(self.logdir))
                    else:
                        logging.info('Succeeded to delete train dir {}.'.format(self.logdir))
                else:
                    # Do nothing.
                    pass

                # Build graph, namely building a graph and initialize member variables associated with graph.
                self.saver = self._build_graph()
            else:
                self.saver = self._restore_graph()

            # After either building a graph or restoring a graph, graph is CONSTRUCTED successfully.
            # Get collections to be used in training.
            self.global_step = tf.get_collection('global_step')[0]
            self.init_op = tf.get_collection('init_op')[0]
            self.train_op = tf.get_collection('train_op')[0]
            self.summary_op = tf.get_collection('summary_op')[0]
            self.raw_features_batch = tf.get_collection('raw_features_batch')[0]
            self.labels_batch = tf.get_collection('labels_batch')[0]
            self.loss = tf.get_collection('loss')[0]
            self.pred_prob = tf.get_collection('predictions')[0]

        if self._check_graph_initialized():
            logging.info('Succeeded to initialize logistic regression Graph.')
        else:
            logging.error('Failed to initialize logistic regression Graph.')

        # Start or restore training.
        # To avoid summary causing memory usage peak, manually save summaries.
        sv = tf.train.Supervisor(graph=self.graph, init_op=self.init_op, logdir=self.logdir,
                                 global_step=self.global_step, summary_op=None,
                                 save_model_secs=600, saver=self.saver)

        with sv.managed_session() as sess:
            logging.info("Entering training loop...")
            for step in range(self.max_train_steps):
                if sv.should_stop():
                    # Save the final model and break.
                    self.saver.save(sess, save_path='{}_{}'.format(sv.save_path, 'final'))
                    break

                if step % 100 == 0:
                    _, summary, global_step_val = sess.run(
                        [self.train_op, self.summary_op, self.global_step])
                    sv.summary_computed(sess, summary, global_step=global_step_val)
                else:
                    sess.run(self.train_op)

            logging.info("Exited training loop.")

        # Session will close automatically when with clause exits.
        # sess.close()
        sv.stop()
