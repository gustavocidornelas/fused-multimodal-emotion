"""
Created on Mon April 8, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf

from parameters.parameters import *
from model.process_data_text import *
from model.model_text import *
from model.evaluate_text import *


if __name__ == '__main__':

    # splitting the data in training and test sets and preparing it to be fed to the model
    data_handler = ProcessDataText(data_path)
    train_text_data, train_labels, test_text_data, test_labels = data_handler.split_train_test(alpha=0.9)
    text_input_len = train_text_data.shape[1]

    # converting the labels to the one-hot format
    train_labels = data_handler.label_one_hot(label=train_labels, num_categories=num_categories)
    test_labels = data_handler.label_one_hot(label=test_labels, num_categories=num_categories)

    # creating training and validation datasets
    train_iterator, val_iterator, text_input, label_batch, handle = data_handler.create_datasets(train_text_data,
                                                                                                 train_labels,
                                                                                                 test_text_data,
                                                                                                 test_labels,
                                                                                                 batch_size, num_epochs)
    # creating the model
    model = TextModel(text_input, label_batch, batch_size, num_categories, learning_rate, data_handler.dict_size,
                      hidden_dim_text, num_layers_text, dr_prob_text)
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

        # creating training and validation handles (to switch between datasets)
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        # loading pre-trained embedding vector to placeholder
        sess.run(model.embedding_init, feed_dict={model.embedding_GloVe: data_handler.get_glove()})

        batch_count = 1

        # feeding the batches to the model
        while True:
            try:
                _, accuracy, loss, summary = sess.run([model.optimizer, model.accuracy, model.loss, model.summary_op],
                                                      feed_dict={handle: train_handle})
                writer_train.add_summary(summary, global_step=model.global_step.eval())

                print('Batch: ' + str(batch_count) + ' Loss: {:.4f}'.format(loss) +
                      ' Training accuracy: {:.4f}'.format(accuracy))
                batch_count += 1

                # evaluating on the validation set every 50 batches
                if batch_count % 50 == 0:
                    evaluator.evaluate_text_model(sess, model, val_iterator, handle, val_handle, writer_val)

            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break




