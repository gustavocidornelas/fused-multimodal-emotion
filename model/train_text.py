"""
Created on Mon April 8, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf

from parameters import *
from process_data_text import *
from model_text import *
from evaluate_text import *


if __name__ == '__main__':

    # splitting the data in training and test sets and preparing it to be fed to the model
    data_handler = ProcessDataText(data_path)
    train_text_data, train_labels, test_text_data, test_labels, val_text_data, val_labels = \
        data_handler.split_train_test(prop_train=0.8, prop_test=0.05)
    text_input_len = train_text_data.shape[1]

    # converting the labels to the one-hot format
    train_labels = data_handler.label_one_hot(label=train_labels, num_categories=num_categories)
    test_labels = data_handler.label_one_hot(label=test_labels, num_categories=num_categories)
    val_labels = data_handler.label_one_hot(label=val_labels, num_categories=num_categories)

    # creating training, testing and validation datasets
    train_iterator, test_iterator, val_iterator, text_input, label_batch, handle = \
        data_handler.create_datasets(train_text_data, tf.cast(train_labels, dtype=tf.float32), test_text_data,
                                     tf.cast(test_labels, dtype=tf.float32), val_text_data,
                                     tf.cast(val_labels, dtype=tf.float32), batch_size, num_epochs)

    # creating the model
    model = TextModel(text_input, label_batch, batch_size, num_categories, learning_rate, data_handler.dict_size,
                      hidden_dim_text, num_layers_text, dr_prob_text, multimodal_model_status)
    model.build_graph()

    # evaluation object
    evaluator = EvaluateText()

    # training the model
    with tf.Session() as sess:
        # initializing the global variables
        sess.run(tf.global_variables_initializer())

        # writing the graph
        writer_train = tf.summary.FileWriter('../graphs/graph_train', sess.graph)
        writer_val = tf.summary.FileWriter('../graphs/graph_val', sess.graph)

        # training loop
        print("Training...")

        # creating training, testing and validation handles (to switch between datasets)
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        # loading pre-trained embedding vector to placeholder
        sess.run(model.embedding_init, feed_dict={model.embedding_GloVe: data_handler.get_glove()})

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
                    val_accuracy = evaluator.evaluate_text_model_val(sess, model, val_iterator, handle, val_handle,
                                                                     writer_val)

                    # saving the best training accuracy so far
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy

            except tf.errors.OutOfRangeError:
                print('End of training')
                print('Best training accuracy: {:.4f}'.format(best_train_accuracy))
                print('Best validation accuracy: {:.4f}'.format(best_val_accuracy))

                # evaluating on the test set
                test_accuracy = evaluator.evaluate_text_model_test(sess, model, test_iterator, handle, test_handle)

                break

        # saving the final text model
        saver = tf.train.Saver()
        saver.save(sess, '../pretrained-models/pt_text_model')




