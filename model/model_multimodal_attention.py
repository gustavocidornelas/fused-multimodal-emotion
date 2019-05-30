"""
Created on Tue May 14, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf
import numpy as np

from model_audio import *

class MultimodalAttentionModel:
    """
        Class that creates the graph for the multimodal model with global attention

    """

    def __init__(self, text_input, label_batch, batch_size, num_categories, learning_rate, dict_size, hidden_dim_text,
                 num_layers_text, dr_prob_text, multimodal_model_status, audio_input, num_filters_audio,
                 filter_lengths_audio, n_pool_audio, audio_len, dr_prob_audio, hidden_dim_audio, num_layers_audio):
        # general
        self.text_input = text_input
        self.audio_input = audio_input
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

        # text recurrent layers
        self.hidden_dim_text = hidden_dim_text
        self.num_layers_text = num_layers_text
        self.dr_prob_text = dr_prob_text

        # output layer
        self.y_labels = []
        self.M = None
        self.b = None

        # audio network
        self.num_filters_audio = num_filters_audio
        self.filter_lengths_audio = filter_lengths_audio
        self.n_pool_audio = n_pool_audio
        self.audio_input_len = audio_len
        self.dr_prob_audio = dr_prob_audio
        self.hidden_dim_audio = hidden_dim_audio
        self.num_layers_audio = num_layers_audio
        self.conv_out_length = int(audio_len / (n_pool_audio[0] * n_pool_audio[1]))

        # attention
        self.W1 = tf.keras.layers.Dense(hidden_dim_text)
        self.W2 = tf.keras.layers.Dense(hidden_dim_text)
        self.V = tf.keras.layers.Dense(1)

    def _create_audio_model(self):
        """
        Creates the audio model in the graph
        """
        print('Creating the audio model...')
        # instantiating object of the class AudioModel
        audio_model = AudioModel(self.audio_input, self.labels, self.batch_size, self.num_categories,
                                 self.learning_rate, self.num_filters_audio, self.filter_lengths_audio,
                                 self.audio_input_len, self.n_pool_audio, self.hidden_dim_audio, self.num_layers_audio,
                                 self.dr_prob_audio)

        # building the audio model's graph
        audio_model._create_conv_layers()
        audio_model._create_recursive_net()

        # getting the audio hidden states for the current batch
        self.audio_hidden_states = tf.stack(audio_model.outputs_enc, axis=2)

    def _create_placeholders(self):
        """
        Creates the placeholder for the pre-trained embedding
        """
        print('Creating placeholders...')
        self.embedding_GloVe = tf.placeholder(tf.float32, shape=[self.dict_size, self.embed_dim],
                                              name='embedding_placeholder')

    def _create_embedding(self):
        """
        Creates the embedding matrix and embeds the input that will be fed to the RNN
        """
        print('Creating embedding...')

        with tf.name_scope('embedding_layer'):
            self.embedding_matrix = tf.Variable(tf.random_normal([self.dict_size, self.embed_dim], mean=0.0,
                                                                 stddev=0.01, dtype=tf.float32, seed=None),
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
        return tf.contrib.rnn.GRUCell(num_units=self.hidden_dim_text)

    def gru_dropout_cell(self):
        """
        Implements a cell instance with dropout wrapper applied

        Returns
        ----------
        (DropoutWrapper object): instance of a GRU cell with the specified dropout probability
        """
        # specified dropout between the layers
        return tf.contrib.rnn.DropoutWrapper(self.gru_cell(), input_keep_prob=self.dr_prob_text,
                                             output_keep_prob=self.dr_prob_text)

    def attention_mechanism(self, text_hidden_state):
        """
        Implements Luong's attention mechanism between the current text model's hidden state and all of the
        audio model hidden states and returns the context vector

        Parameters
        ----------
        text_hidden_state (tensor): tensor of shape [batch_size, hidden_dim_text] with the current hidden states of the
                                    text model for the batch

        Returns
        ----------
        transformed_hidden_state (tensor): tensor of shape [batch_size, hidden_dim_text] with the transformed hidden
                                           state (after applying attention)
        """
        # reshaping the tensors
        audio_hidden_states = tf.reshape(self.audio_hidden_states, [-1, self.conv_out_length, self.hidden_dim_audio])
        text_hidden_state = tf.reshape(text_hidden_state, [-1, 1, self.hidden_dim_text])

        # calculating the scores (similarity between the current text hidden state and all of the audio hidden states)
        #score = tf.matmul(audio_hidden_states, text_hidden_state)
        score = self.V(tf.nn.tanh(self.W1(audio_hidden_states) + self.W2(text_hidden_state)))

        # calculating the attention weights
        attention_weights = tf.nn.softmax(score, dim=1, name='attention_weights')

        # performing the weighted sum of the audio hidden states
        weighted_states = tf.multiply(audio_hidden_states, attention_weights)
        context_vector = tf.reduce_sum(weighted_states, axis=1)

        # concatenating the context vector with the current text hidden state
        text_hidden_state = tf.reshape(text_hidden_state, [-1, self.hidden_dim_text])
        concatenated_state = tf.concat([context_vector, text_hidden_state], axis=1)

        # projecting back to the same dimensions as the hidden states
        transformed_hidden_state = tf.tanh(tf.matmul(concatenated_state, self.W_c))

        return transformed_hidden_state

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
            cell_enc = tf.nn.rnn_cell.MultiRNNCell([self.gru_dropout_cell() for _ in range(self.num_layers_text)],
                                                   state_is_tuple=False)

            # initializing list with the outputs
            self.outputs_enc = []
            self.W_c = tf.Variable(tf.random_uniform([self.hidden_dim_audio + self.hidden_dim_text,
                                                      self.hidden_dim_text],
                                                     minval=-0.25,
                                                     maxval=0.25,
                                                     dtype=tf.float32,
                                                     seed=None
                                                     ),
                                   trainable=True,
                                   name='project_weight')

            # unrolling the RNN manually
            for time_step, input in enumerate(rnn_input):

                # checking if it is the first time step
                if time_step == 0:
                    # initialize hidden state of the text RNN with the last state from the audio RNN
                    output, text_hidden_state = cell_enc(input, self.audio_hidden_states[:, :, -1])
                    self.outputs_enc.append(output)
                else:
                    # get the transformed hidden state with attention
                    new_hidden_state = self.attention_mechanism(text_hidden_state)

                    # feed the transformed hidden state to the RNN
                    output, text_hidden_state = cell_enc(input, new_hidden_state)
                    self.outputs_enc.append(output)

            # adding hidden states to collection to be restored later
            hidden_states = tf.stack(self.outputs_enc, axis=2)
            tf.add_to_collection('hidden_states', hidden_states)

            # self.final_encoder = self.last_states_enc[-1]
            self.final_encoder = text_hidden_state

    def _create_output_layers(self):
        """
        Creates the output layer (fully connected layer)
        """
        print('Creating the output layers...')

        # defining the output layer
        with tf.name_scope('output_layer'):
            self.M = tf.Variable(tf.random_uniform([self.hidden_dim_text, self.num_categories],
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
                                           tf.argmax(self.labels, 1)), tf.float32), name='mean_batch_accuracy')

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
        Method that builds the graph for the multimodal model with attention
        """
        self._create_audio_model()
        self._create_placeholders()
        self._create_embedding()
        self._initialize_embedding()
        self._create_recursive_net()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()
