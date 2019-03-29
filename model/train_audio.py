"""
Created on Mon March 25, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf

from parameters.parameters import *
from model.process_data_audio import *
from model.model_audio import *


if __name__ == '__main__':

    # splitting the data in training and test sets and preparing it to be fed to the model
    data_handler = ProcessDataAudio(data_path)
    train_audio_data, train_labels, test_audio_data, test_labels = data_handler.split_train_test(alpha=0.9)
    # audio_input = data_handler.get_batches(train_audio_data, batch_size)
    audio_input = train_audio_data

    # creating the model (tensorflow graph)
    model = AudioModel(batch_size, num_categories, learning_rate, num_filters, filter_lengths, audio_input.shape[1], n_pool,
                       encoder_size, hidden_dim, num_layers)
    model.build_graph()

    # visualizing the graph (just to check)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('../graphs', sess.graph)
