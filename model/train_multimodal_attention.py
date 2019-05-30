"""
Created on Tue May 14, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf

import gc

from parameters import *
from process_data_text import *
from process_data_audio import *
from process_data_multimodal import *
from model_text import *
from model_multimodal_attention import *
from evaluate_multimodal_attention import *


if __name__ == '__main__':

    # instantiating all of the data handlers
    text_data_handler = ProcessDataText(data_path)
    audio_data_handler = ProcessDataAudio(data_path)
    multi_data_handler = ProcessDataMultimodal(data_path, text_data_handler, audio_data_handler)

    del audio_data_handler
    gc.collect()

    # splitting the data int training, validation and test sets
    train_text_data, train_audio_data, train_labels, test_text_data, test_audio_data, test_labels, val_text_data, \
    val_audio_data,  val_labels = multi_data_handler.split_train_test(prop_train=0.8, prop_test=0.05)

    # converting the labels to the one-hot format
    train_labels = text_data_handler.label_one_hot(label=train_labels, num_categories=num_categories)
    test_labels = text_data_handler.label_one_hot(label=test_labels, num_categories=num_categories)
    val_labels = text_data_handler.label_one_hot(label=val_labels, num_categories=num_categories)

    # creating the text datasets
    text_placeholder = tf.placeholder(tf.int32, shape=[None, train_text_data.shape[1]], name='text_input_placeholder')
    audio_placeholder = tf.placeholder(tf.float32, shape=[None, train_audio_data.shape[1]],
                                       name='audio_input_placeholder')
    label_placeholder = tf.placeholder(tf.float32, shape=[None, num_categories], name='labels_placeholder')

    train_iterator, test_iterator, val_iterator, text_input, audio_input, label_batch, handle = \
        multi_data_handler.create_datasets(text_placeholder, audio_placeholder, label_placeholder, test_text_data,
                                           test_audio_data, test_labels, tf.cast(val_text_data, dtype=tf.int32),
                                           tf.cast(val_audio_data, dtype=tf.float32),
                                           tf.cast(val_labels, dtype=tf.float32), batch_size, num_epochs)

    del multi_data_handler
    gc.collect()

    # creating the multimodal model with attention
    multimodal_model = MultimodalAttentionModel(text_input, label_batch, batch_size, num_categories, learning_rate,
                                                text_data_handler.dict_size, hidden_dim_text, num_layers_text,
                                                dr_prob_text, multimodal_model_status, audio_input, num_filters_audio,
                                                filter_lengths_audio, n_pool_audio, train_audio_data.shape[1],
                                                dr_prob_audio, hidden_dim_audio, num_layers_audio)
    multimodal_model.build_graph()

    # evaluation object
    evaluator = EvaluateMultimodalAttention()

    # training the model
    with tf.Session() as sess:
        # initializing the global variables
        sess.run(tf.global_variables_initializer())

        # writing the graph
        writer_train = tf.summary.FileWriter('../graphs/graph_train', sess.graph)
        writer_val = tf.summary.FileWriter('../graphs/graph_val', sess.graph)

        # training loop
        print("Training...")

        # initializing iterator with the training data
        sess.run(train_iterator.initializer, feed_dict={text_placeholder: train_text_data,
                                                        audio_placeholder: train_audio_data,
                                                        label_placeholder: train_labels})

        # creating the text training, testing and validation handles (to switch between datasets)
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        # loading pre-trained embedding vector to placeholder
        sess.run(multimodal_model.embedding_init, feed_dict={multimodal_model.embedding_GloVe:
                                                             text_data_handler.get_glove()})

        batch_count = 1

        # keeping track of the best test and validation accuracies
        best_train_accuracy = 0
        best_val_accuracy = 0

        del text_data_handler
        gc.collect()

        # feeding the batches to the model
        while True:
            try:
                _, accuracy, loss, summary = sess.run(
                    [multimodal_model.optimizer, multimodal_model.accuracy, multimodal_model.loss,
                     multimodal_model.summary_op],
                    feed_dict={handle: train_handle})

                writer_train.add_summary(summary, global_step=multimodal_model.global_step.eval())

                print('Batch: ' + str(batch_count) + ' Loss: {:.4f}'.format(loss) +
                      ' Training accuracy: {:.4f}'.format(accuracy))

                # saving the best training accuracy so far
                if accuracy > best_train_accuracy:
                    best_train_accuracy = accuracy

                batch_count += 1

                # evaluating on the validation set every 50 batches
                if batch_count % 50 == 0:
                    # calculating the accuracy on the validation set
                    val_accuracy = evaluator.evaluate_multi_model_val(sess, multimodal_model, val_iterator, handle,
                                                                      val_handle, writer_val)

                    # saving the best training accuracy so far
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy

            except tf.errors.OutOfRangeError:
                print('End of training')
                print('Best training accuracy: {:.4f}'.format(best_train_accuracy))
                print('Best validation accuracy: {:.4f}'.format(best_val_accuracy))

                # evaluating on the test set
                test_accuracy = evaluator.evaluate_multi_model_test(sess, multimodal_model, test_iterator, handle,
                                                                   test_handle)

                break

        # saving the final text model
        saver = tf.train.Saver()
        saver.save(sess, '../pretrained-models/pt_multimodal_model')