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
    audio_input_len = train_audio_data.shape[1]

    # converting the labels to the one-hot format
    train_labels = data_handler.label_one_hot(label=train_labels, num_categories=num_categories)
    test_labels = data_handler.label_one_hot(label=test_labels, num_categories=num_categories)

    # generating the batches
    dataset, iterator = data_handler.get_batches(train_audio_data, train_labels, batch_size, num_epochs)

    # getting next batch
    audio_input, label_batch = iterator.get_next()

    # creating the model
    model = AudioModel(audio_input, label_batch, batch_size, num_categories, learning_rate, num_filters, filter_lengths,
                       audio_input_len, n_pool, encoder_size, hidden_dim, num_layers)
    model.build_graph()

    # training the model
    with tf.Session() as sess:
        # initializing the global variables
        sess.run(tf.global_variables_initializer())

        # writing the graph
        writer = tf.summary.FileWriter('../graphs', sess.graph)

        # training loop
        print("Training...")
        for i in range(num_training_steps):
            print('Iteration ' + str(i))
            _, summary = sess.run([model.optimizer, model.summary_op])
            writer.add_summary(summary, global_step=model.global_step.eval())

