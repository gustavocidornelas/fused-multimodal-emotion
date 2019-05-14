"""
Created on Fri May 10, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf

from parameters.parameters import *
from model.process_data_text import *
from model.process_data_audio import *
from model.process_data_multimodal import *
from model.model_text import *
from model.import_model import *
from model.evaluate_text import *


if __name__ == '__main__':

    # instantiating all of the data handlers
    text_data_handler = ProcessDataText(data_path)
    audio_data_handler = ProcessDataAudio(data_path)
    multi_data_handler = ProcessDataMultimodal(data_path, text_data_handler, audio_data_handler)

    # splitting the data int training, validation and test sets
    train_text_data, train_audio_data, train_labels, test_text_data, test_audio_data, test_labels, val_text_data, \
    val_audio_data,  val_labels = multi_data_handler.split_train_test(prop_train=0.8, prop_test=0.05)

    # converting the labels to the one-hot format
    train_labels = text_data_handler.label_one_hot(label=train_labels, num_categories=num_categories)
    test_labels = text_data_handler.label_one_hot(label=test_labels, num_categories=num_categories)
    val_labels = text_data_handler.label_one_hot(label=val_labels, num_categories=num_categories)

    # creating the text datasets
    train_text_iterator, test_text_iterator, val_text_iterator, text_input, text_label_batch, text_handle = \
        text_data_handler.create_datasets(train_text_data, train_labels, test_text_data, test_labels, val_text_data,
                                     val_labels, batch_size, num_epochs)

    # creating the text model (model that is going to be trained)
    text_model = TextModel(text_input, text_label_batch, batch_size, num_categories, learning_rate,
                           text_data_handler.dict_size, hidden_dim_text, num_layers_text, dr_prob_text,
                           multimodal_model_status)
    text_model.build_graph()

    # evaluation object
    evaluator = EvaluateText()

    # training the model
    with tf.Session() as sess:
        # initializing the global variables
        sess.run(tf.global_variables_initializer())

        # writing the graph
        writer_train = tf.summary.FileWriter('../graphs/graph_train', sess.graph)
        writer_val = tf.summary.FileWriter('../graphs/graph_val', sess.graph)

        # importing the pre-trained audio model
        audio_model = ImportAudioModel(train_audio_data, train_labels, test_audio_data, test_labels, val_audio_data, val_labels)

        # training loop
        print("Training...")

        # creating the text training, testing and validation handles (to switch between datasets)
        train_text_handle = sess.run(train_text_iterator.string_handle())
        test_text_handle = sess.run(test_text_iterator.string_handle())
        val_text_handle = sess.run(val_text_iterator.string_handle())

        # loading pre-trained embedding vector to placeholder
        sess.run(text_model.embedding_init, feed_dict={text_model.embedding_GloVe: text_data_handler.get_glove()})

        batch_count = 1

        # keeping track of the best test and validation accuracies
        best_train_accuracy = 0
        best_val_accuracy = 0

        # feeding the batches to the model
        while True:
            try:
                hidden_states = audio_model.run_audio_model_train()
                initial_hidden_states = hidden_states[:, :, -1]

                _, accuracy, loss, summary = sess.run([text_model.optimizer, text_model.accuracy, text_model.loss,
                                                       text_model.summary_op],
                                                      feed_dict={text_handle: train_text_handle,
                                                                 text_model.initial_hidden_state: initial_hidden_states})

                writer_train.add_summary(summary, global_step=text_model.global_step.eval())

                print('Batch: ' + str(batch_count) + ' Loss: {:.4f}'.format(loss) +
                      ' Training accuracy: {:.4f}'.format(accuracy))

                # saving the best training accuracy so far
                if accuracy > best_train_accuracy:
                    best_train_accuracy = accuracy

                batch_count += 1

                # evaluating on the validation set every 50 batches
                #if batch_count % 50 == 0:
                    # calculating the accuracy on the validation set
                #    val_accuracy = evaluator.evaluate_text_model_val(sess, text_model, val_text_iterator, text_handle,
                 #                                                    val_text_handle, writer_val)

                    # saving the best training accuracy so far
                 #   if val_accuracy > best_val_accuracy:
                 #       best_val_accuracy = val_accuracy

            except tf.errors.OutOfRangeError:
                print('End of training')
                print('Best training accuracy: {:.4f}'.format(best_train_accuracy))
                print('Best validation accuracy: {:.4f}'.format(best_val_accuracy))

                # evaluating on the test set
                #test_accuracy = evaluator.evaluate_text_model_test(sess, text_model, test_text_iterator, text_handle,
                 #                                                  test_text_handle)

                break

