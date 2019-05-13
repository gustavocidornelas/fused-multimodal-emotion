"""
Created on Mon April 8, 2019

@author: Gustavo Cid Ornelas
"""
import tensorflow as tf


class TextModel:
    """
        Class that creates the text model graph

    """

    def __init__(self, text_input, label_batch, batch_size, num_categories, learning_rate, dict_size, hidden_dim,
                 num_layers, dr_prob, multimodal_model_status):
        # general
        self.text_input = text_input
        self.labels = label_batch
        self.batch_size = batch_size
        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.loss = 0.0
        self.batch_loss = None
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.multimodal_model_status = multimodal_model_status

        # embedding layer
        self.dict_size = dict_size
        self.embed_dim = 300  # using GloVe

        # recurrent layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dr_prob = dr_prob

        # output layers
        self.y_labels = []
        self.M = None
        self.b = None

    def _create_placeholders(self):
        """
        Creates the placeholder for the pre-trained embedding
        """
        print('Creating placeholders...')
        self.embedding_GloVe = tf.placeholder(tf.float64, shape=[self.dict_size, self.embed_dim],
                                              name='embedding_placeholder')
        self.initial_hidden_state = tf.placeholder(tf.float64, shape=[None, self.hidden_dim],
                                                   name='initial_rnn_hidden_state')

    def _create_embedding(self):
        """
        Creates the embedding matrix and embeds the input that will be fed to the RNN
        """
        print('Creating embedding...')

        with tf.name_scope('embedding_layer'):
            self.embedding_matrix = tf.Variable(tf.random_normal([self.dict_size, self.embed_dim], mean=0.0,
                                                                 stddev=0.01, dtype=tf.float64, seed=None),
                                                trainable=True, name='embed_matrix')

            self.embedded_input = tf.nn.embedding_lookup(self.embedding_matrix, self.text_input, name='embedded_input')

    def _initialize_embedding(self):
        """
        Initializes the embedding matrix with the pre-trained embedding (GloVe)
        """
        print('Initializing embedding with GloVe...')

        self.embedding_init = self.embedding_matrix.assign(self.embedding_GloVe)

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
            # splitting the embedded input to be fed to the RNN as a list
            rnn_input = tf.split(self.embedded_input, 128, axis=1)

            # reshaping the elements in rnn_input to be fed to the RNN
            rnn_input = [input_element[:, 0, :] for input_element in rnn_input]

            # creating the list with the specified number of layers of GRU cells with dropout
            cell_enc = tf.nn.rnn_cell.MultiRNNCell([self.gru_dropout_cell() for _ in range(self.num_layers)],
                                                   state_is_tuple=False)

            if self.multimodal_model_status:
                # simulating the time steps in the RNN with initialized hidden state
                self.outputs_enc, self.last_states_enc = tf.nn.static_rnn(cell=cell_enc, inputs=rnn_input,
                                                                          initial_state=self.initial_hidden_state,
                                                                          dtype=tf.float64)
            else:
                # simulating the time steps in the RNN without hidden state initialization
                self.outputs_enc, self.last_states_enc = tf.nn.static_rnn(cell=cell_enc, inputs=rnn_input,
                                                                          dtype=tf.float64)

            # adding hidden states to collection to be restored later
            hidden_states = tf.stack(self.outputs_enc, axis=2)
            tf.add_to_collection('hidden_states', hidden_states)

            #self.final_encoder = self.last_states_enc[-1]
            self.final_encoder = self.last_states_enc

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
                                                   dtype=tf.float64,
                                                   seed=None),
                                 trainable=True, name='W')

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float64), trainable=True, name='b')

            self.batch_prediction = tf.add(tf.matmul(self.final_encoder, self.M), self.b, name='batch_prediction')

        with tf.name_scope('loss'):
            #  batch loss
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_prediction, labels=self.labels)
            self.loss = tf.reduce_mean(self.batch_loss, name='mean_batch_loss')

            # batch accuracy
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.batch_prediction, 1),
                                           tf.argmax(self.labels, 1)), tf.float64), name='mean_batch_accuracy')

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
        Creating the TensorBoard summary. Displays the mean training loss and mean training accuracy
        """
        print('Creating summary...')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            tf.summary.scalar('mean_accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
        Method that builds the graph for the text model, which consists of recurrent layers
        """
        self._create_placeholders()
        self._create_embedding()
        self._initialize_embedding()
        self._create_recursive_net()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()
