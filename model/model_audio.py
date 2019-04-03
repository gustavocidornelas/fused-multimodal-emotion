"""
Created on Mon March 25, 2019

@author: Gustavo Cid Ornelas
"""
import tensorflow as tf
import numpy as np

class AudioModel:
    """
    Class that creates the audio model (CNN + GRU) graph

    """
    def __init__(self, audio_input, label_batch, batch_size, num_categories, learning_rate, num_filters, filter_lengths,
                 audio_len, n_pool, encoder_size, hidden_dim, num_layers):

        # general
        self.audio_input = tf.reshape(audio_input, shape=[-1, audio_len, 1])
        self.labels = label_batch
        self.batch_size = batch_size
        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.loss = 0.0
        self.batch_loss = None
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # convolutional layers
        self.num_filters = num_filters
        self.filter_lengths = filter_lengths
        self.len_audio = audio_len
        self.n_pool = n_pool
        self.conv_out_length = int(audio_len / (n_pool[0] * n_pool[1]))

        # recurrent layers
        self.encoder_size = encoder_size # may be useless
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dr_prob = tf.placeholder(tf.float32, name='dropout_prob')

        # output layers
        self.y_labels = []
        self.M = None
        self.b = None

    def _create_conv_layers(self):
        """
        Builds the convolutional layers of the audio model

        """
        print('Creating the convolutional layers...')

        with tf.name_scope('CNN'):
            # first convolutional layer
            conv_layer1 = tf.layers.conv1d(self.audio_input, filters=self.num_filters[0],
                                           kernel_size=self.filter_lengths[0], padding='same', activation=tf.nn.relu)

            # max pooling across time
            max_pool1 = tf.layers.max_pooling1d(conv_layer1, pool_size=self.n_pool[0], strides=self.n_pool[0])

            # second convolutional layer
            conv_layer2 = tf.layers.conv1d(max_pool1, filters=self.num_filters[1],
                                           kernel_size=self.filter_lengths[1], padding='same', activation=tf.nn.relu)

            # max pooling across time
            max_pool2 = tf.layers.max_pooling1d(conv_layer2, pool_size=self.n_pool[1], strides=self.n_pool[1])

            # max pooling across channels
            self.output_cnn = tf.reduce_max(max_pool2, reduction_indices=[2], keepdims=True)

    def gru_cell(self):
        """
        Creates an instance of a Gated Recurrent Unit (GRU) cell

        Returns
        ----------
        (GRUCell object): instance of a GRU cell
        """
        # a single instance of the GRU
        return tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)

    def gru_dropout_cell(self):
        """
        Implements a cell instance with dropout wrapper applied

        Returns
        ----------
        (DropoutWrapper object): instance of a GRU cell with the specified dropout probability
        """
        # specified dropout between the layers
        return tf.contrib.rnn.DropoutWrapper(self.gru_cell(), input_keep_prob=0.5,
                                             output_keep_prob=0.5)

    def _create_recursive_net(self):
        """
        Creates the RNN with GRUs and dropout, as specified
        """
        print('Creating the recurrent layers...')

        with tf.name_scope('RNN'):
            # first, we split the output of the CNN to be fed as a sequence to the RNN
            # TODO: fix this to work with the full tensor
            rnn_input = tf.split(self.output_cnn[:, :, 0], self.conv_out_length, axis=1)

            # creating the list with the specified number of layers of GRU cells with dropout
            cell_enc = tf.nn.rnn_cell.MultiRNNCell([self.gru_dropout_cell() for _ in range(self.num_layers)])

            # simulating the time steps in the RNN (returns output activations and last hidden state)
            self.outputs_enc, last_states_enc = tf.nn.static_rnn(cell=cell_enc, inputs=rnn_input, dtype=tf.float64)

            self.final_encoder = last_states_enc[-1]

    def _create_output_layers(self):
        """
        Creates the output layer, which is a multi-layer perceptron
        """
        print('Creating the output layers...')

        # defining the output layer
        with tf.name_scope('output_layer'):
            self.M = tf.Variable(tf.random_uniform([self.hidden_dim, self.num_categories],
                                                   minval=-0.25,
                                                   maxval=0.25,
                                                   dtype=tf.float64,
                                                   seed=None),
                                 trainable=True, name='W')

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float64), trainable=True, name='b')

            self.batch_prediction = tf.matmul(self.final_encoder, self.M) + self.b

        with tf.name_scope('loss'):
            #  batch loss
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_prediction, labels=self.labels)
            self.loss = tf.reduce_mean(self.batch_loss)

            # batch accuracy
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.batch_prediction, 1),
                                          tf.argmax(self.labels, 1)), tf.float64))/self.batch_size

    def _create_optimizer(self):
        """
        Defining the optimizer (Adam)
        """
        print('Creating the optimizer...')

        with tf.name_scope('audio_optimizer'):
            # Adam optimizer with the specified learning rate
            opt_func = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gvs = opt_func.compute_gradients(self.loss)
            # applying gradient clipping
            gvs_cap = [(tf.clip_by_value(t=grad, clip_value_min=-10.0, clip_value_max=10.0), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=gvs_cap, global_step=self.global_step)

    def _create_summary(self):
        """
        Creating the TensorBoard summary. Displays the mean loss
        """
        print('Creating summary...')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            tf.summary.scalar('mean_accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
        Method that builds the graph for the audio model, which consists of convolutional layers followed by recurrent
        layers
        """
        self._create_conv_layers()
        self._create_recursive_net()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()

