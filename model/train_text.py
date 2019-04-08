"""
Created on Mon April 8, 2019

@author: Gustavo Cid Ornelas
"""

import tensorflow as tf

from parameters.parameters import *
from model.process_data_text import *
from model.model_text import *


if __name__ == '__main__':

    # splitting the data in training and test sets and preparing it to be fed to the model
    data_handler = ProcessDataText(data_path)
    train_text_data, train_labels, test_text_data, test_labels = data_handler.split_train_test(alpha=0.9)
    text_input_len = train_text_data.shape[1]

    # converting the labels to the one-hot format
    train_labels = data_handler.label_one_hot(label=train_labels, num_categories=num_categories)
    test_labels = data_handler.label_one_hot(label=test_labels, num_categories=num_categories)

    with tf.name_scope('dataset'):
        # creating the data placeholders
        text_placeholder = tf.placeholder(tf.int32, shape=[None, text_input_len])
        label_placeholder = tf.placeholder(tf.float64, shape=[None, num_categories])

        # creating dataset over the placeholders
        dataset = tf.data.Dataset.from_tensor_slices((text_placeholder, label_placeholder))
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        # creating iterator
        iterator = dataset.make_initializable_iterator()

        text_input, label_batch = iterator.get_next()

    # creating the model
    model = TextModel(text_input, label_batch, batch_size, num_categories, learning_rate, data_handler.dict_size,
                      hidden_dim_text, num_layers_text, dr_prob_text)
    model.build_graph()

    # training the model
    with tf.Session() as sess:
        # initializing the global variables
        sess.run(tf.global_variables_initializer())

        # writing the graph
        writer = tf.summary.FileWriter('../graphs', sess.graph)

        # training loop
        print("Training...")

        # initializing iterator with training data
        sess.run(iterator.initializer, feed_dict={text_placeholder: train_text_data, label_placeholder: train_labels})

        # loading pre-trained embedding vector to placeholder
        sess.run(model.embedding_init, feed_dict={model.embedding_GloVe: data_handler.get_glove()})

        count = 1

        # feeding the batches to the model
        while True:
            try:
                _, accuracy, loss, summary = sess.run([model.optimizer, model.accuracy, model.loss, model.summary_op])
                writer.add_summary(summary, global_step=model.global_step.eval())

                print('Batch: ' + str(count) + ' Loss: {:.4f}'.format(loss) +
                      ' Training accuracy: {:.4f}'.format(accuracy))
                count += 1

            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
