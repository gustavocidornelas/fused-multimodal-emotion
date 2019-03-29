"""
Created on Mon March 25, 2019

@author: Gustavo Cid Ornelas
"""
import tensorflow as tf


class AudioModel:
    """
    Class that contains the audio model

    """
    def __init__(self, batch_size, num_categories, learning_rate, num_filters, filter_lengths, audio_len, n_pool,
                 encoder_size, hidden_dim, num_layers):

        # general
        self.batch_size = batch_size
        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.loss = 0
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # convolutional layers
        self.num_filters = num_filters
        self.filter_lengths = filter_lengths
        self.len_audio = audio_len
        self.n_pool = n_pool
        self.conv_out_length = int(audio_len / n_pool)

        # recurrent layers
        self.encoder_size = encoder_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # output layers
        self.y_labels = []
        self.M = None
        self.b = None

    def _create_placeholders(self):
        """
            Creates all of the placeholders for the graph
        """

        print('Creating the placeholders...')

        with tf.name_scope('audio_placeholder'):
            # general placeholders
            self.y_labels = tf.placeholder(tf.float32, shape=[None, self.num_categories], name='label')

            # placeholders for the convolutional layers
            self.audio_input = tf.placeholder(tf.float32, shape=[None, self.len_audio, 1], name='audio')

            # placeholders for the recurrent layers
            self.dr_prob = tf.placeholder(tf.float32, name='dropout_prob')

    def _create_conv_layers(self):
        """
        Builds the convolutional layers of the audio model

        """

        print('Creating the convolutional layers...')

        # first convolutional layer
        conv_layer1 = tf.layers.conv1d(self.audio_input, filters=self.num_filters[0],
                                       kernel_size=self.filter_lengths[0], padding='same', activation=tf.nn.relu)

        # second convolutional layer
        conv_layer2 = tf.layers.conv1d(conv_layer1, filters=self.num_filters[1],
                                       kernel_size=self.filter_lengths[1], padding='same', activation=tf.nn.relu)

        # max pooling across time
        self.output_cnn = tf.layers.max_pooling1d(conv_layer2, pool_size=self.n_pool, strides=self.n_pool)

        # max pooling across channels
        self.output_cnn = tf.reduce_max(self.output_cnn, reduction_indices=[2], keepdims=True)

    def gru_cell(self):
        """
        Creates an instance of a Gated Recurrent Unit (GRU) cell

        Returns
        ----------
        (tensor)
        """

        # a single instance of the GRU
        return tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)

    def gru_dropout_cell(self):
        """
        Implements a cell instance with dropout wrapper applied

        Returns
        ----------

        """
        # specified dropout between the layers
        return tf.contrib.rnn.DropoutWrapper(self.gru_cell(), input_keep_prob=self.dr_prob,
                                             output_keep_prob=self.dr_prob)

    def _create_recursive_net(self):
        """
        Creates the RNN with GRUs and dropout, as specified
        """

        print('Creating the recurrent layers...')

        #with tf.name_scope('audio_RNN'):
        #    with tf.variable_scope('audio_GRU', reuse=False, initializer=tf.orthogonal_initializer()):
        # first, we split the output of the CNN to be fed as a sequence to the RNN
        # TODO: fix this to work with the full tensor
        rnn_input = tf.split(self.output_cnn[:, :, 0], self.conv_out_length, axis=1)

        # creating the list with the specified number of layers of GRU cells with dropout
        cell_enc = tf.nn.rnn_cell.MultiRNNCell([self.gru_dropout_cell() for _ in range(self.num_layers)])

        # simulating the time steps in the RNN (returns output activations and last hidden state)
        self.outputs_enc, last_states_enc = tf.nn.static_rnn(cell=cell_enc, inputs=rnn_input, dtype=tf.float32)

        self.final_encoder = last_states_enc[-1]

    def _create_output_layers(self):
        """
        Creates the output layer, which is a multi-layer perceptron
        """
        print('Creating the output layers...')

        self.M = tf.Variable(tf.random_uniform([self.hidden_dim, self.num_categories],
                                               minval=-0.25,
                                               maxval=0.25,
                                               dtype=tf.float32,
                                               seed=None),
                             trainable=True, name='weights_matrix')

        self.b = tf.Variable(tf.zeros([1], dtype=tf.float32), trainable=True, name='output_bias')

        self.batch_prediction = tf.matmul(self.final_encoder, self.M) + self.b

    def _create_optimizer(self):
        """
        Defining the optimizer (Adam)
        """

        print('Creating the optimizer...')

        with tf.name_scope('audio_optimizer'):
            opt_func = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gvs = opt_func.compute_gradients(self.loss)
            #vs_cap = [(tf.clip_by_value(t=grad, clip_value_min=-10.0, clip_value_max=10.0), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=gvs, global_step=self.global_step) # edit to gvs_cap

    def _create_summary(self):

        print('Creating summary...')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
        Method that builds the graph for the audio model, which consists of convolutional layers followed by recurrent
        layers
        """

        self._create_placeholders()
        self._create_conv_layers()
        self._create_recursive_net()
        self._create_output_layers()
        #self._create_optimizer()
        self._create_summary()

