import numpy as np
import tensorflow as tf
from tensorflow import logging

from constants import make_summary
from read_data import get_input_data_tensors


class LinearClassifier(object):
    def __init__(self, logdir='/tmp/linear_reg'):
        """
        Args:
             logdir: Path to the log dir.
        """
        self.logdir = logdir
        self.weights = None
        self.biases = None
        self.rmse = np.NINF

    def fit(self, data_pipeline, raw_feature_size, tr_data_fn=None, tr_data_paras=None,
            l2_regs=None, validate_set=None, line_search=True):
        """
        Compute weights and biases of linear classifier using normal equation.
            With line search for best l2_reg.
        Args:
            data_pipeline: A namedtuple consisting of the following elements.
                reader, features reader.
                data_pattern, File Glob of data set.
                batch_size, How many examples to handle per time.
                num_threads, How many IO threads to prefetch examples.
            raw_feature_size: The dimensionality of features.
            tr_data_fn: a function that transforms input data.
            tr_data_paras: Other parameters should be passed to tr_data_fn. A dictionary.
            l2_regs: A list, each element represents how much the linear classifier weights should be penalized.
            validate_set: (data, labels) with dtype float32.
                The data set (numpy arrays) used to choose the best l2_reg.
                Sampled from whole validate set if necessary.
            line_search: Boolean argument representing whether to do boolean search.

        Returns: Weights and biases fit on the given data set, where biases are appended as the last row.

        """
        logging.info('Entering linear classifier ...')

        reader = data_pipeline.reader
        batch_size = data_pipeline.batch_size
        num_classes = reader.num_classes
        # Transform raw_feature_size to a list.
        if isinstance(raw_feature_size, int):
            raw_feature_size = [raw_feature_size]
        else:
            raw_feature_size = list(raw_feature_size)
        logging.info('Linear regression uses {}-dimensional features.'.format(raw_feature_size))

        feature_size = raw_feature_size

        if line_search:
            # Both l2_regs and validate_set are required.
            if l2_regs is None:
                raise ValueError('There is no l2_regs to do line search.')
            else:
                logging.info('l2_regs is {}.'.format(l2_regs))

            if validate_set is None:
                raise ValueError('There is no validate_set to do line search for l2_reg.')
            else:
                validate_data, validate_labels = validate_set
        else:
            # Simply fit the training set. Make l2_regs have only one element.
            if l2_regs is None:
                l2_regs = [0.00001]
            elif isinstance(l2_regs, float):
                l2_regs = [l2_regs]
            elif isinstance(l2_regs, list) or isinstance(l2_regs, tuple):
                l2_regs = l2_regs[:1]
            logging.info('No line search, l2_regs is {}.'.format(l2_regs))
            if validate_set is None:
                # Important! To make the graph construction successful.
                validate_data = np.zeros([1] + raw_feature_size, dtype=np.float32)
                validate_labels = np.zeros([1, num_classes], dtype=np.float32)
            else:
                validate_data, validate_labels = validate_set

        # Check validate data and labels shape.
        logging.info('validate set: data has shape {}, labels has shape {}.'.format(
            validate_data.shape, validate_labels.shape))

        if list(validate_data.shape[1:]) != raw_feature_size:
            raise ValueError('validate set shape does not conforms with training set.')

        # TO BE CAUTIOUS! THE FOLLOWING MAY HAVE TO DEAL WITH FEATURE SIZE CHANGE.
        # Check extra data transform function arguments.
        # If transform changes the features size, change it.
        if tr_data_fn is not None:
            if tr_data_paras is None:
                tr_data_paras = {}
            else:
                if ('reshape' in tr_data_paras) and (tr_data_paras['reshape'] is True):
                    feature_size = tr_data_paras['size']
                    logging.warn('Data transform changes the features size to {}.'.format(feature_size))

        # Method - append an all-one col to X by using block matrix multiplication (all-one col is treated as a block).
        # Create the graph to traverse all data once.
        with tf.Graph().as_default() as graph:
            global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
            global_step_inc_op = tf.assign_add(global_step, 1)

            # X.transpose * X
            norm_equ_1_initializer = tf.placeholder(tf.float32, shape=[feature_size, feature_size])
            norm_equ_1 = tf.Variable(initial_value=norm_equ_1_initializer, collections=[], name='X_Tr_X')

            # X.transpose * Y
            norm_equ_2_initializer = tf.placeholder(tf.float32, shape=[feature_size, num_classes])
            norm_equ_2 = tf.Variable(initial_value=norm_equ_2_initializer, collections=[], name='X_Tr_Y')

            example_count = tf.Variable(initial_value=0.0, name='example_count')
            features_sum = tf.Variable(initial_value=tf.zeros([feature_size]), name='features_sum')
            labels_sum = tf.Variable(initial_value=tf.zeros([num_classes]), name='labels_sum')

            # label is one-hot encoded, int32 type.
            id_batch, raw_features_batch, labels_batch = get_input_data_tensors(
                data_pipeline, onehot_label=True, num_epochs=1, name_scope='input')
            if tr_data_fn is None:
                tr_features_batch = tf.identity(raw_features_batch)
            else:
                tr_features_batch = tr_data_fn(raw_features_batch, **tr_data_paras)

            with tf.name_scope('batch_increment'):
                tr_features_batch_tr = tf.matrix_transpose(tr_features_batch, name='X_Tr')
                float_labels_batch = tf.to_float(labels_batch)
                batch_norm_equ_1 = tf.matmul(tr_features_batch_tr, tr_features_batch,
                                             name='batch_norm_equ_1')
                # batch_norm_equ_1 = tf.add_n(tf.map_fn(lambda x: tf.einsum('i,j->ij', x, x),
                #                                       transformed_features_batch_tr), name='X_Tr_X')
                batch_norm_equ_2 = tf.matmul(tr_features_batch_tr, float_labels_batch,
                                             name='batch_norm_equ_2')
                batch_example_count = tf.cast(tf.shape(tr_features_batch)[0], tf.float32,
                                              name='batch_example_count')
                batch_features_sum = tf.reduce_sum(tr_features_batch, axis=0,
                                                   name='batch_features_sum')
                batch_labels_sum = tf.reduce_sum(float_labels_batch, axis=0,
                                                 name='batch_labels_sum')

            with tf.name_scope('update_ops'):
                update_norm_equ_1_op = tf.assign_add(norm_equ_1, batch_norm_equ_1)
                update_norm_equ_2_op = tf.assign_add(norm_equ_2, batch_norm_equ_2)
                update_example_count = tf.assign_add(example_count, batch_example_count)
                update_features_sum = tf.assign_add(features_sum, batch_features_sum)
                update_labels_sum = tf.assign_add(labels_sum, batch_labels_sum)

            with tf.control_dependencies([update_norm_equ_1_op, update_norm_equ_2_op,
                                          update_example_count, update_features_sum,
                                          update_labels_sum, global_step_inc_op]):
                update_equ_non_op = tf.no_op(name='unified_update_op')

            with tf.name_scope('solution'):
                # After all data being handled, compute weights.
                l2_reg_ph = tf.placeholder(tf.float32, shape=[])
                l2_reg_term = tf.diag(tf.fill([feature_size], l2_reg_ph), name='l2_reg')
                # X.transpose * X + lambda * Id, where d is the feature dimension.
                norm_equ_1_with_reg = tf.add(norm_equ_1, l2_reg_term)

                # Concat other blocks to form the final norm equation terms.
                final_norm_equ_1_top = tf.concat([norm_equ_1_with_reg, tf.expand_dims(features_sum, 1)], 1)
                final_norm_equ_1_bot = tf.concat([features_sum, tf.expand_dims(example_count, 0)], 0)
                final_norm_equ_1 = tf.concat([final_norm_equ_1_top, tf.expand_dims(final_norm_equ_1_bot, 0)], 0,
                                             name='norm_equ_1')
                final_norm_equ_2 = tf.concat([norm_equ_2, tf.expand_dims(labels_sum, 0)], 0,
                                             name='norm_equ_2')

                # The last row is the biases.
                weights_biases = tf.matrix_solve(final_norm_equ_1, final_norm_equ_2, name='weights_biases')

                weights = weights_biases[:-1]
                biases = weights_biases[-1]

            with tf.name_scope('loss'):
                predictions = tf.matmul(tr_features_batch, weights) + biases

                squared_loss = tf.reduce_sum(
                    tf.squared_difference(predictions, float_labels_batch), name='squared_loss')
                # pred_labels = tf.greater_equal(predictions, 0.0, name='pred_labels')

            summary_op = tf.summary.merge_all()

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                               name='init_glo_loc_var')

            phase_train_pls = tf.get_collection('phase_train_pl')
            phase_train_pl = phase_train_pls[0] if len(phase_train_pls) > 0 else None
            val_feed_dict = {phase_train_pl: False} if phase_train_pl is not None else {}

        sess = tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False))
        # Initialize variables.
        sess.run(init_op)
        sess.run([norm_equ_1.initializer, norm_equ_2.initializer], feed_dict={
            norm_equ_1_initializer: np.zeros([feature_size, feature_size], dtype=np.float32),
            norm_equ_2_initializer: np.zeros([feature_size, num_classes], dtype=np.float32)
        })

        # If logdir does not exist, it will be created automatically.
        summary_writer = tf.summary.FileWriter(self.logdir, graph=sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _, summary, global_step_val = sess.run([update_equ_non_op, summary_op, global_step],
                                                       feed_dict=val_feed_dict)
                summary_writer.add_summary(summary, global_step=global_step_val)

                print('Done step {}.'.format(global_step_val))
        except tf.errors.OutOfRangeError:
            logging.info('Finished normal equation terms computation -- one epoch done.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            summary_writer.close()

        # Wait for threads to finish.
        coord.join(threads)

        # Line search.
        best_weights_val, best_biases_val = None, None
        best_l2_reg = 0
        min_loss = np.PINF

        for l2_reg in l2_regs:
            # Compute regularized weights.
            weights_val, biases_val = sess.run([weights, biases], feed_dict={l2_reg_ph: l2_reg})
            # Compute validation loss.
            num_validate_videos = validate_data.shape[0]
            split_indices = np.linspace(0, num_validate_videos + 1,
                                        num=max(num_validate_videos // batch_size + 1, 2),
                                        dtype=np.int32)
            loss_vals = []
            for i in range(len(split_indices) - 1):
                start_ind = split_indices[i]
                end_ind = split_indices[i + 1]

                par_val_data = validate_data[start_ind:end_ind]
                # One-hot encoded labels.
                par_val_labels = np.eye(num_classes)[validate_labels[start_ind:end_ind]]

                # Avoid re-computing weights and biases (Otherwise, l2_reg_ph is necessary).
                # ith_loss_val is the total squared loss of the batch.
                ith_loss_val = sess.run(squared_loss,
                                        feed_dict={
                                            raw_features_batch: par_val_data,
                                            float_labels_batch: par_val_labels,
                                            weights: weights_val,
                                            biases: biases_val})

                loss_vals.append(ith_loss_val)

            validate_loss_val = sum(loss_vals) / num_validate_videos

            logging.info('l2_reg {} leads to rmse loss {}.'.format(l2_reg, validate_loss_val))
            if validate_loss_val < min_loss:
                best_weights_val, best_biases_val = weights_val, biases_val
                min_loss = validate_loss_val
                best_l2_reg = l2_reg

        if (not line_search) and (validate_set is None):
            min_loss = None

        sess.close()

        logging.info('The best l2_reg is {} with rmse loss {}.'.format(best_l2_reg, min_loss))
        logging.info('Exiting linear classifier ...')

        self.weights = best_weights_val
        self.biases = best_biases_val
        self.rmse = min_loss


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
        self.tr_data_paras = None
        self.multi_label = None
        self.init_learning_rate = None
        self.decay_steps = None
        self.decay_rate = None
        self.epochs = None
        self.l1_reg_rate = None
        self.l2_reg_rate = None
        # positive / negative class weight.
        # Used in binary classification, incl. one-vs-rest multi-label.
        self.pos_weights = None
        # Used to initialize the soft-max layer.
        self.initial_weights = None
        self.initial_biases = None

        # Whether to use a pretrained model
        self.use_pretrain = False
        self.pretrained_model_dir = None
        self.pretrained_scope = None
        self.pretrained_var_list = None
        self.pretrained_saver = None

        self.graph = None
        # Member variables associated with graph.
        self.global_step = None
        self.init_op = None
        # Training op to optimize the loss with respect to the soft-max layer.
        self.train_op_w = None
        # Training op to optimize the loss with respect to the whole network.
        self.train_op = None
        self.summary_op = None
        self.saver = None
        self.raw_features_batch = None
        self.labels_batch = None
        self.loss = None
        self.pred_prob = None
        self.pred_labels = None
        self.phase_train_pl = None

    @staticmethod
    def average_gradients(tower_grads):
        """
        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
        Returns:
          List of pairs of (gradient, variable) where the gradient has been averaged
          across all towers.
        Reference:
            cifar10_multi_gpu_train.py in tensorflow/models.
        """
        if len(tower_grads) == 1:
            return tower_grads[0]
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = [g for g, _ in grad_and_vars]

            mean_grad = tf.scalar_mul(1.0 / len(grad_and_vars), tf.accumulate_n(grads))

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            mean_grad_and_var = (mean_grad, v)
            average_grads.append(mean_grad_and_var)
        return average_grads

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

        # Define the last classification layer, softmax or multi-label classification.
        with tf.variable_scope('Classification', reuse=None):
            if self.initial_weights is None:
                weights = tf.get_variable('weights', shape=[self.feature_size, self.num_classes],
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / np.sqrt(self.feature_size)),
                                          regularizer=tf.identity)
            else:
                weights = tf.get_variable('weights', shape=[self.feature_size, self.num_classes],
                                          initializer=self.initial_weights,
                                          regularizer=tf.identity)

            if self.initial_biases is None:
                biases = tf.get_variable('biases', shape=[self.num_classes],
                                         initializer=tf.constant_initializer(0.001))
            else:
                biases = tf.get_variable('biases', shape=[self.num_classes],
                                         initializer=self.initial_biases)

        # Stage 1, tune softmax layer only when restoring from a pretrained checkpoint.
        optimizer_w = tf.train.RMSPropOptimizer(learning_rate=self.init_learning_rate,
                                                name='opt_softmax')

        # Stage 2, tune the whole net.
        # Fine tuning the transformation and softmax layer with AdamOptimizer
        # Note sharing the same optimizer is not recommended, for they use info of previous gradients.
        optimizer = tf.train.AdamOptimizer(learning_rate=self.init_learning_rate / 10.0,
                                           name='opt_full_net')

        # Get training data, multi-label
        id_batch, raw_features_batch, labels_batch = get_input_data_tensors(
            self.train_data_pipeline, onehot_label=self.multi_label,
            shuffle=True, num_epochs=self.epochs, name_scope='Input')

        with tf.name_scope('Split'):
            num_parallelism = 1
            raw_features_batch_splits = tf.split(raw_features_batch, num_parallelism, axis=0)
            labels_batch_splits = tf.split(labels_batch, num_parallelism, axis=0)

        with tf.name_scope('Loss'):
            # Split the training batch into smaller mini-batches.
            tower_losses = []
            tower_pred_prob, tower_pred_labels = [], []
            tower_gradients_w, tower_gradients = [], []
            for i in range(num_parallelism):
                # Perform the feature transformation.
                if self.tr_data_fn is None:
                    i_tr_features = tf.identity(raw_features_batch_splits[i])
                else:
                    i_tr_features = self.tr_data_fn(raw_features_batch_splits[i],
                                                    **{**self.tr_data_paras,
                                                       'reuse': True if i > 0 else None})
                    # Get the pretrained variables just after creating the transformation!!!
                    # Later operations (e.g., RMSPROP) might add extra variables to the same scope.
                    # This will cause an error while restoring the variables.
                    if self.use_pretrain is True and self.pretrained_var_list is None:
                        self.pretrained_var_list = self.graph.get_collection(
                            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.pretrained_scope) + self.graph.get_collection(
                            tf.GraphKeys.LOCAL_VARIABLES, scope=self.pretrained_scope)
                        logging.debug('pretrained_var_list has {} variables'.format(len(self.pretrained_var_list)))

                i_logits = tf.add(tf.matmul(i_tr_features, weights), biases, name='logits_{}'.format(i+1))
                i_pred_prob = tf.nn.softmax(i_logits, dim=-1, name='pred_probability_{}'.format(i+1))
                i_pred_labels = tf.argmax(i_logits, axis=-1, name='pred_labels_{}'.format(i+1))

                tower_pred_prob.append(i_pred_prob)
                tower_pred_labels.append(i_pred_labels)

                if self.multi_label:
                    # multi-label classification
                    i_logistic_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.to_float(labels_batch_splits[i]),
                        logits=i_logits, name='logistic_loss_{}'.format(i + 1))
                    i_loss_per_example = tf.reduce_sum(
                        i_logistic_loss, axis=-1, name='loss_per_example_{}'.format(i + 1))
                else:
                    i_loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels_batch_splits[i], logits=i_logits, name='x_entropy_per_example')

                i_loss = tf.reduce_mean(i_loss_per_example, name='mean_loss_{}'.format(i+1))

                tower_losses.append(tf.expand_dims(i_loss, 0))
                # compute_gradients only return the gradients over given var_list.
                if self.use_pretrain:
                    tower_gradients_w.append(optimizer_w.compute_gradients(
                        i_loss, var_list=[weights, biases],
                        aggregation_method=tf.AggregationMethod.DEFAULT)
                    )
                tower_gradients.append(optimizer.compute_gradients(
                    i_loss,
                    aggregation_method=tf.AggregationMethod.DEFAULT)
                )

            loss = tf.reduce_mean(tf.concat(tower_losses, 0), axis=0, name='loss')
            pred_prob = tf.concat(tower_pred_prob, 0, name='pred_probability')
            pred_labels = tf.concat(tower_pred_labels, 0, name='pred_labels')

        with tf.name_scope('Regularization'):
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
                tf.summary.scalar('reg_loss', reg_loss)
            else:
                reg_loss = tf.constant(0.0, name='zero_reg_loss')

        with tf.variable_scope('Train'):
            # TODO, Compute reg_loss in terms of variables, _w and full network
            if self.use_pretrain:
                train_op_w = optimizer_w.apply_gradients(
                    self.average_gradients(tower_gradients_w), global_step=global_step)
            else:
                train_op_w = tf.no_op('NO_TRAIN_OP_W')
            train_op = optimizer.apply_gradients(
                self.average_gradients(tower_gradients), global_step=global_step)

        # Add summary to trainable variables.
        with tf.name_scope('Summary'):
            for variable in tf.trainable_variables():
                if len(variable.shape) == 0:
                    tf.summary.scalar(variable.op.name, variable)
                else:
                    tf.summary.histogram(variable.op.name, variable)

        summary_op = tf.summary.merge_all()
        # summary_op = tf.constant(1.0)

        # num_epochs needs local variables to be initialized.
        # Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Used for restoring training checkpoints.
        tf.add_to_collection('global_step', global_step)
        tf.add_to_collection('init_op', init_op)
        tf.add_to_collection('train_op_w', train_op_w)
        tf.add_to_collection('train_op', train_op)
        tf.add_to_collection('summary_op', summary_op)
        # Add to collection. In inference, get collection and feed it with test data.
        tf.add_to_collection('raw_features_batch', raw_features_batch)
        tf.add_to_collection('labels_batch', labels_batch)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('pred_prob', pred_prob)
        tf.add_to_collection('pred_labels', pred_labels)

        # To save global variables and savable objects, i.e., var_list is None.
        # Using rbf transform will also save centers and scaling factors.
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=0.5)

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
            graph_ops = [self.global_step, self.init_op, self.train_op, self.train_op_w,
                         self.summary_op, self.saver, self.raw_features_batch, self.labels_batch,
                         self.loss, self.pred_prob, self.pred_labels]

            return (self.graph is not None) and (graph_ops.count(None) == 0)

    # Used for restoring pretrained model
    def _load_pre_train_model(self):
        if self.use_pretrain:
            # Load pre-trained graph.
            self.pretrained_checkpoint = tf.train.latest_checkpoint(self.pretrained_model_dir)
            logging.info("Pretrained checkpoint: {}".format(self.pretrained_checkpoint))
            self.pretrained_saver = tf.train.Saver(var_list=self.pretrained_var_list,
                                                   name='pretrained_saver')
            # The inline function takes a session object as input.
            return lambda s: self.pretrained_saver.restore(s, self.pretrained_checkpoint)
        else:
            return None

    def fit(self, train_data_pipeline, raw_feature_size, start_new_model=False,
            tr_data_fn=None, tr_data_paras=None, multi_label=False,
            validation_set=None, validation_fn=None,
            init_learning_rate=0.0001, decay_steps=40000, decay_rate=0.95, epochs=None,
            l1_reg_rate=None, l2_reg_rate=None, pos_weights=None,
            initial_weights=None, initial_biases=None,
            use_pretrain=False, pretrained_model_dir=None, pretrained_scope=None):
        """
        Logistic regression fit function.
        Args:
            train_data_pipeline: A namedtuple consisting of reader, data_pattern, batch_size and num_threads.
            raw_feature_size: The dimensionality of features.
            start_new_model: If True, start a new model instead of restoring from existing checkpoints.
            tr_data_fn: a function that transforms input data.
            tr_data_paras: Other parameters should be passed to tr_data_fn. A dictionary.
            multi_label: If use multi-label classification or multi-class classification.
            validation_set: A tuple contains the validation features and labels.
            validation_fn: The function to compute validation metric.
            init_learning_rate: Decayed gradient descent parameter.
            decay_steps: Decayed gradient descent parameter.
            decay_rate: Decayed gradient descent parameter.
            epochs: Maximal epochs to use.
            l1_reg_rate: None, not impose l1 regularization.
            l2_reg_rate: l2 regularization rate.
            pos_weights: For imbalanced binary classes. Here, num_pos << num_neg, the weights should be > 1.0.
                If None, treated as 1.0 for all binary classifiers.
            initial_weights: If not None, the weights will be initialized with it.
            initial_biases: If not None, the biases will be initialized with it.
            use_pretrain: Whether to use pretrained model.
            pretrained_model_dir: The folder that contains the pretrained model.
            pretrained_scope: The scope to hold the pretrained model.
        Returns: None.
        """
        reader = train_data_pipeline.reader
        batch_size = train_data_pipeline.batch_size
        num_classes = reader.num_classes

        self.train_data_pipeline = train_data_pipeline
        self.raw_feature_size = raw_feature_size
        self.feature_size = self.raw_feature_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.tr_data_fn = tr_data_fn
        self.tr_data_paras = tr_data_paras
        self.multi_label = multi_label
        self.init_learning_rate = init_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.l1_reg_rate = l1_reg_rate
        self.l2_reg_rate = l2_reg_rate
        self.pos_weights = pos_weights
        self.initial_weights = initial_weights
        self.initial_biases = initial_biases

        self.use_pretrain = use_pretrain
        self.pretrained_model_dir = pretrained_model_dir
        self.pretrained_scope = pretrained_scope

        # Check extra data transform function arguments.
        # If transform changes the features size, change it.
        if isinstance(self.tr_data_paras, dict):
            if ('reshape' in self.tr_data_paras) and (self.tr_data_paras['reshape'] is True):
                self.feature_size = self.tr_data_paras['size']
                logging.warn('Data transform changes the features size to {}.'.format(
                    self.feature_size))
                logging.debug('Data transform arguments are {}.'.format(self.tr_data_paras))
        else:
            self.tr_data_paras = dict()

        logging.info('Logistic regression uses {}-dimensional features.'.format(self.feature_size))

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

                # Build graph.
                self.saver = self._build_graph()
            else:
                # Restore from a checkpoint.
                self.saver = self._restore_graph()

            # After either building a graph or restoring a graph, graph is CONSTRUCTED successfully.
            # Get collections to be used in training.
            self.global_step = tf.get_collection('global_step')[0]
            self.init_op = tf.get_collection('init_op')[0]
            self.train_op_w = tf.get_collection('train_op_w')[0]
            self.train_op = tf.get_collection('train_op')[0]
            self.summary_op = tf.get_collection('summary_op')[0]
            self.raw_features_batch = tf.get_collection('raw_features_batch')[0]
            self.labels_batch = tf.get_collection('labels_batch')[0]
            self.loss = tf.get_collection('loss')[0]
            self.pred_prob = tf.get_collection('pred_prob')[0]
            self.pred_labels = tf.get_collection('pred_labels')[0]

            phase_train_pls = tf.get_collection('phase_train_pl')
            self.phase_train_pl = phase_train_pls[0] if len(phase_train_pls) > 0 else None

            train_feed_dict = {self.phase_train_pl: True} if self.phase_train_pl is not None else {}
            val_feed_dict = {self.phase_train_pl: False} if self.phase_train_pl is not None else {}

        if self._check_graph_initialized():
            logging.info('Succeeded to initialize logistic regression Graph.')
        else:
            logging.error('Failed to initialize logistic regression Graph.')

        # Clean validation set
        val_data, val_labels = None, None
        if validation_set is not None:
            val_data, val_labels = validation_set
            # Cut until the multiples of batch_size.
            num_val_images = (len(val_labels) // self.batch_size) * self.batch_size
            if num_val_images > 0:
                val_data, val_labels = val_data[:num_val_images], val_labels[:num_val_images]
                # multi-label classification requires onehot-encoded labels
                if self.multi_label is True:
                    eye_mat = np.eye(num_classes)
                    val_labels = eye_mat[val_labels]
            else:
                logging.warn('Not enough validation data {} < batch size {}.'.format(
                    len(val_labels), self.batch_size))
        else:
            logging.info('No validation set is found.')

        # Start or restore training.
        # To avoid summary causing memory usage peak, manually save summaries.
        sv = tf.train.Supervisor(graph=self.graph, init_op=self.init_op, logdir=self.logdir,
                                 global_step=self.global_step, summary_op=None,
                                 save_model_secs=1800, saver=self.saver,
                                 init_fn=self._load_pre_train_model())

        # allow_soft_placement must be set to True to build towers on GPU,
        # as some of the ops do not have GPU # implementations.
        with sv.managed_session(
                config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)) as sess:
            logging.info("Entering training loop...")
            # Obtain the current training step. Continue training from a checkpoint.
            start_step = sess.run(self.global_step)
            for step in range(start_step, self.max_train_steps):
                if sv.should_stop():
                    break

                # Train the softmax layer for 10000 steps and then train the full network.
                if self.use_pretrain and step <= 10000:
                    current_train_op = self.train_op_w
                    # Don't use dropout nor update batch normalization.
                    current_train_feed_dict = val_feed_dict
                else:
                    current_train_op = self.train_op
                    current_train_feed_dict = train_feed_dict

                if step % 400 == 0:
                    if step == 400:
                        # Only record once
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                    else:
                        run_options = None
                        run_metadata = None

                    _, summary, loss_val, global_step_val = sess.run(
                        [current_train_op, self.summary_op, self.loss, self.global_step],
                        feed_dict=current_train_feed_dict,
                        options=run_options, run_metadata=run_metadata)

                    if run_metadata is not None:
                        sv.summary_writer.add_run_metadata(run_metadata,
                                                           'step_{}'.format(step),
                                                           global_step=global_step_val)
                    # Add train summary.
                    sv.summary_computed(sess, summary, global_step=global_step_val)
                    # Add training loss summary.
                    print('Step {}, training loss {:.6f}'.format(global_step_val, loss_val))

                    if step % 800 == 0:
                        # Compute validation loss and performance (validation_fn).
                        if num_val_images > 0:
                            # Compute validation loss.
                            split_indices = np.linspace(0, num_val_images + 1,
                                                        num=max(num_val_images // self.batch_size + 1, 2),
                                                        dtype=np.int32)

                            val_loss_vals, val_pred_labels = [], []
                            for i in range(len(split_indices) - 1):
                                start_ind = split_indices[i]
                                end_ind = split_indices[i + 1]

                                if validation_fn is not None:
                                    ith_val_loss_val, ith_pred_labels = sess.run(
                                        [self.loss, self.pred_labels], feed_dict={
                                            **val_feed_dict,
                                            self.raw_features_batch: val_data[start_ind:end_ind],
                                            self.labels_batch: val_labels[start_ind:end_ind]})

                                    val_pred_labels.extend(ith_pred_labels)
                                else:
                                    ith_val_loss_val = sess.run(
                                        self.loss, feed_dict={
                                            **val_feed_dict,
                                            self.raw_features_batch: val_data[start_ind:end_ind],
                                            self.labels_batch: val_labels[start_ind:end_ind]})

                                val_loss_vals.append(ith_val_loss_val * (end_ind - start_ind))

                            val_loss_val = sum(val_loss_vals) / num_val_images
                            # Add validation summary.
                            sv.summary_writer.add_summary(
                                make_summary('validation/xentropy', val_loss_val), global_step_val)

                            if validation_fn is not None:
                                val_func_name = validation_fn.__name__
                                val_per = validation_fn(predictions=val_pred_labels, labels=val_labels)

                                sv.summary_writer.add_summary(
                                    make_summary('validation/{}'.format(val_func_name), val_per),
                                    global_step_val)
                                print('Step {}, validation loss: {}, {}: {}.'.format(
                                    global_step_val, val_loss_val, val_func_name, val_per))
                            else:
                                print('Step {}, validation loss: {}.'.format(
                                    global_step_val, val_loss_val))
                else:
                    sess.run(current_train_op, feed_dict=current_train_feed_dict)

        logging.info("Exited training loop.")

        # Session will close automatically when with clause exits.
        # sess.close()
        sv.stop()
