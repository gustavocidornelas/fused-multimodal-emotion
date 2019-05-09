"""
Created on Fri March 22, 2019

@author: Gustavo Cid Ornelas
"""
import numpy as np
import random
import tensorflow as tf


class ProcessDataAudio:
    """
    Deals with the audio data. Loads all of the audio data and labels and splits it into training and testing sets

    Attributes
    ----------
    audio_data (array): array of shape [num_samples, 250.000]. Each row corresponds to a raw audio file
    labels (array): array of shape [num_samples] corresponding to the categories
    """
    def __init__(self, data_path):

        self.data_path = data_path  # e. g. '../data/processed-data/'

        # loading the audio data and the labels
        self.audio_data = self._load_audio()
        self.labels = self._load_labels()

    def _load_audio(self):
        """
        Loads the audio FC_raw_audio.npy file  and creates an array with all of the audio samples

        Returns
        ----------
        audio_data (array): array of shape [num_samples, 250.000]. Each row corresponds to a raw audio file
        """
        print('Loading the audio data...')
        audio_file = self.data_path + 'FC_raw_audio.npy'

        # reading the audio file
        audio_data = np.load(audio_file)

        return audio_data

    def _load_labels(self):
        """
        Loads the audio FC_label.txt file  and creates an array with all of the labels

        Returns
        ----------
        labels (array): array of shape [num_samples] corresponding to the categories
        """
        print('Loading the labels...')

        labels_file = self.data_path + 'FC_label.txt'

        # reading the labels file
        labels = np.genfromtxt(labels_file, delimiter=',')

        return labels

    def split_train_test(self, prop_train, prop_test):
        """
        Splits the data into training and testing sets in the proportion defined by alpha

        Parameters
        ----------
        prop_train (float): number between 0 and 1 that determines the proportion of the data that will be used for
                            training
        prop_test (float): number between 0 and 1 that determines the proportion of the data that will be used for
                            testing
        Returns
        ----------
        train_audio_data, test_audio_data, val_audio_data (array): arrays that correspond to the raw audio samples
        train_labels, test_labels, val_labels (array): arrays with the labels in the same order as the audio_data
        """
        print('Splitting the data ...')

        # setting the seed
        random.seed(31)

        # total number of samples in the dataset
        num_samples = len(self.labels)

        if prop_train < 0 or prop_train > 1 or prop_test < 0 or prop_test > 1:
            raise ValueError('Inserted proportion for either training or testing is out of range. It must be between 0 '
                             'and 1')

        # number of elements in each set
        num_train = int(prop_train * num_samples)
        num_test = int(prop_test * num_samples)

        # concatenating data and labels to shuffle
        data = np.concatenate((self.audio_data, self.labels.reshape(num_samples, 1)), axis=1)

        # shuffling the data
        np.random.shuffle(data)

        # splitting the data
        train_audio_data = data[:num_train + 1, :-1]
        train_labels = data[:num_train + 1, -1]
        test_audio_data = data[num_train + 1: num_train + 1 + num_test + 1, :-1]
        test_labels = data[num_train + 1: num_train + 1 + num_test + 1, -1]
        val_audio_data = data[num_train + 1 + num_test + 1:, :-1]
        val_labels = data[num_train + 1 + num_test + 1:, -1]

        return train_audio_data, train_labels, test_audio_data, test_labels, val_audio_data, val_labels

    def label_one_hot(self, label, num_categories):
        """
        Converts the labels to the format one-hot, which is expected by our model

        Returns
        ----------
        one_hot_label (array): array of shape [len(label), num_categories] with a 1 in the corresponding category
        """
        print('Converting the labels to one-hot format...')

        # array that stores all of the one-hot vectors for the labels [samples, num_categories])
        one_hot_label = np.zeros((len(label), num_categories))

        for idx, element in enumerate(label):
            one_hot_label[idx, int(element)] = 1

        return one_hot_label

    def create_datasets(self, audio_placeholder, label_placeholder, test_audio_data, test_labels, val_audio_data,
                        val_labels, batch_size, num_epochs):
        """
        Creates the training and validation datasets and returns the iterators, next elements of the dataset and the
        handle

        Parameters
        ----------
        train_audio_data (array): array of shape [num_train_samples, 250.000] with training samples as rows
        train_labels (array): array of shape [num_train_samples, num_categories] labels for training
        test_audio_data (array): array of shape [num_test_samples, 250.000] with training samples as rows
        test_labels (array): array of shape [num_test_samples, num_categories] labels for training
        val_audio_data (array): array of shape [num_val_samples, 250.000] with validation samples as rows
        val_labels (array): array of shape [num_val_samples, num_categories] labels for validation
        batch_size (int): batch size (for the training set)
        num_epochs (int): number of epochs

        Returns
        ----------
        train_iterator (Iterator object): iterator for the training dataset
        test_iterator (Iterator object): iterator for the test dataset
        val_iterator (Iterator object): iterator for the validation dataset
        audio_input (tensor): tensor with the text inputs to be fed to the model
        label_batch (tensor): tensor with the labels to be fed to the model
        handle (string): handle string, to switch between the datasets
        """
        with tf.name_scope('dataset'):

            # creating the training dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((audio_placeholder, label_placeholder))
            train_dataset = train_dataset.repeat(num_epochs)
            train_dataset = train_dataset.batch(batch_size)

            # creating the test dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((test_audio_data, test_labels))
            test_dataset = test_dataset.batch(test_audio_data.shape[0])

            # creating the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((val_audio_data, val_labels))
            val_dataset = val_dataset.batch(val_audio_data.shape[0])

            # creating the iterators from the datasets
            train_iterator = train_dataset.make_initializable_iterator()
            test_iterator = test_dataset.make_initializable_iterator()
            val_iterator = val_dataset.make_initializable_iterator()

            # creating the handle
            handle = tf.placeholder(tf.string, shape=[], name='handle')

            # creating iterator
            iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                           train_dataset.output_shapes)

            # getting the next element
            audio_input, label_batch = iterator.get_next()

        return train_iterator, test_iterator, val_iterator, audio_input, label_batch, handle



