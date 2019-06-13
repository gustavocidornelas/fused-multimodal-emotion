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
                 audio_len, n_pool, hidden_dim, num_layers, dr_prob):

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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dr_prob = dr_prob

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
            self.output_cnn = max_pool2

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
        return tf.contrib.rnn.DropoutWrapper(self.gru_cell(), input_keep_prob=self.dr_prob,
                                             output_keep_prob=self.dr_prob)

    def _create_recursive_net(self):
        """
        Creates the RNN with GRUs and dropout, as specified
        """
        print('Creating the recurrent layers...')

        with tf.name_scope('RNN'):
            # first, we split the output of the CNN to be fed as a sequence to the RNN
            rnn_input = tf.split(self.output_cnn, self.conv_out_length, axis=1)

            # reshaping the elements in rnn_input to be fed to the RNN
            rnn_input = [input_element[:, 0, :] for input_element in rnn_input]

            # creating the list with the specified number of layers of GRU cells with dropout
            cell_enc = tf.nn.rnn_cell.MultiRNNCell([self.gru_dropout_cell() for _ in range(self.num_layers)])

            # simulating the time steps in the RNN (returns output activations and last hidden state)
            self.outputs_enc, last_states_enc = tf.nn.static_rnn(cell=cell_enc, inputs=rnn_input, dtype=tf.float32)

            self.final_encoder = last_states_enc[-1]

            # adding hidden states to collection to be restored later
            hidden_states = tf.stack(self.outputs_enc, axis=2)
            tf.add_to_collection('hidden_states', hidden_states)

    def _create_output_layers(self):
        """
        Creates the output layer (fully connected layer)
        """
        print('Creating the output layers...')

        # defining the output layer
        with tf.name_scope('output_layer'):
            self.M = tf.Variable(tf.random_uniform([self.hidden_dim, self.num_categories],
                                                   minval=-0.25,
                                                   maxval=0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                 trainable=True, name='W')

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32), trainable=True, name='b')

            self.batch_prediction = tf.add(tf.matmul(self.final_encoder, self.M), self.b, name='batch_prediction')

        with tf.name_scope('loss'):
            #  batch loss
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_prediction, labels=self.labels)
            self.loss = tf.reduce_mean(self.batch_loss, name='mean_batch_loss')

            # batch accuracy
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.batch_prediction, 1),
                                                            tf.argmax(self.labels, 1)), tf.float32),
                                           name='mean_batch_accuracy')

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

