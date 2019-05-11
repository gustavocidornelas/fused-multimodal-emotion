"""
Created on Fri May 10, 2019

@author: Gustavo Cid Ornelas
"""
import tensorflow as tf

from parameters.parameters import *


class ImportAudioModel:
    """
    Class that restores the graph of the trained model specified in '../pretrained-models' and runs it locally to return
    the values of interest
    """
    def __init__(self, train_audio_data, train_labels):
        # creating a local graph
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            # restoring the audio model graph
            print('Restoring the audio model...')
            saver = tf.train.import_meta_graph('../pretrained-models/pt_audio_model.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint('../pretrained-models/'))

            # creating the audio datasets
            self._create_audio_datasets()

            self.predictions = self.graph.get_tensor_by_name('output_layer/batch_prediction:0')
            self.hidden_states = self.graph.get_collection('hidden_states')[0]

            # getting the handle
            self.handle = self.graph.get_tensor_by_name('dataset/handle:0')

            # initializing the global variables
            self.sess.run(tf.global_variables_initializer())

            # initializing the training audio dataset with the audio data
            self.sess.run(self.train_audio_iterator.initializer, feed_dict={self.audio_input_placeholder:
                                                                                train_audio_data,
                                                                            self.labels_placeholder: train_labels})

            # creating the audio training, testing and validation handles (to switch between datasets)
            self.train_audio_handle = self.sess.run(self.train_audio_iterator.string_handle())
            #self.test_audio_handle = self.sess.run(test_audio_iterator.string_handle())
            #self.val_audio_handle = self.sess.run(val_audio_iterator.string_handle())

    def _create_audio_datasets(self):

        # recovering the placeholders
        self.audio_input_placeholder = self.graph.get_tensor_by_name('audio_input_placeholder:0')
        self.labels_placeholder = self.graph.get_tensor_by_name('labels_placeholder:0')

        # training dataset
        self.train_audio_dataset = tf.data.Dataset.from_tensor_slices((self.audio_input_placeholder,
                                                                       self.labels_placeholder))
        self.train_audio_dataset = self.train_audio_dataset.repeat(num_epochs)
        self.train_audio_dataset = self.train_audio_dataset.batch(batch_size)
        self.train_audio_iterator = self.train_audio_dataset.make_initializable_iterator()

    def run_audio_model_train(self):

        print('Feeding batch to audio model...')
        batch_prediction, batch_hidden_states = self.sess.run([self.predictions, self.hidden_states],
                                                              feed_dict={self.handle: self.train_audio_handle})

        return batch_hidden_states



