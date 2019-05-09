"""
Created on Mon March 25, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf
import os

from parameters.parameters import *
from model.process_data_audio import *
from model.model_audio import *
from model.evaluate_audio import *


if __name__ == '__main__':

    # splitting the data in training and test sets and preparing it to be fed to the model
    data_handler = ProcessDataAudio(data_path)
    train_audio_data, train_labels, test_audio_data, test_labels, val_audio_data, val_labels = \
        data_handler.split_train_test(prop_train=0.8, prop_test=0.05)
    audio_input_len = train_audio_data.shape[1]

    # converting the labels to the one-hot format
    train_labels = data_handler.label_one_hot(label=train_labels, num_categories=num_categories)
    test_labels = data_handler.label_one_hot(label=test_labels, num_categories=num_categories)
    val_labels = data_handler.label_one_hot(label=val_labels, num_categories=num_categories)

    # placeholders
    audio_placeholder = tf.placeholder(tf.float64, shape=[None, audio_input_len], name='audio_input_placeholder')
    label_placeholder = tf.placeholder(tf.float64, shape=[None, num_categories], name='labels_placeholder')

    # creating training and validation datasets
    train_iterator, test_iterator, val_iterator, audio_input, label_batch, handle = data_handler.create_datasets(
        audio_placeholder, label_placeholder, test_audio_data, test_labels, val_audio_data, val_labels, batch_size,
        num_epochs)

    # creating the model
    model = AudioModel(audio_input, label_batch, batch_size, num_categories, learning_rate, num_filters_audio,
                       filter_lengths_audio, audio_input_len, n_pool_audio, hidden_dim_audio, num_layers_audio,
                       dr_prob_audio)
    model.build_graph()

    # evaluation object
    evaluator = EvaluateAudio()

    # training the model
    with tf.Session() as sess:
        # initializing the global variables
        sess.run(tf.global_variables_initializer())

        # writing the graphs
        writer_train = tf.summary.FileWriter('../graphs/graph_train', sess.graph)
        writer_val = tf.summary.FileWriter('../graphs/graph_val', sess.graph)

        # training loop
        print("Training...")

        # initializing iterator with training data
        sess.run(train_iterator.initializer, feed_dict={audio_placeholder: train_audio_data, label_placeholder:
                 train_labels})

        # creating training and validation handles (to switch between datasets)
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        batch_count = 1

        # keeping track of the best test and validation accuracies
        best_train_accuracy = 0
        best_val_accuracy = 0

        # feeding the batches to the model
        while True:
            try:
                _, accuracy, loss, summary = sess.run([model.optimizer, model.accuracy, model.loss, model.summary_op],
                                                      feed_dict={handle: train_handle})
                writer_train.add_summary(summary, global_step=model.global_step.eval())

                print('Batch: ' + str(batch_count) + ' Loss: {:.4f}'.format(loss) +
                      ' Training accuracy: {:.4f}'.format(accuracy))

                # saving the best training accuracy so far
                if accuracy > best_train_accuracy:
                    best_train_accuracy = accuracy

                batch_count += 1

                # evaluating on the validation set every 50 batches
                if batch_count % 50 == 0:
                    # calculating the accuracy on the validation set
                    val_accuracy = evaluator.evaluate_audio_model_val(sess, model, val_iterator, handle, val_handle,
                                                                      writer_val)

                    # saving the best training accuracy so far
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy

            except tf.errors.OutOfRangeError:
                print('End of dataset')
                print('Best training accuracy: {:.4f}'.format(best_train_accuracy))
                print('Best validation accuracy: {:.4f}'.format(best_val_accuracy))

                # evaluating on the test set
                test_accuracy = evaluator.evaluate_audio_model_test(sess, model, test_iterator, handle, test_handle)

                break

        # saving the trained audio model
        print('Saving the trained model...')
        saver = tf.train.Saver()
        saver.save(sess, 'pt_audio_model')

