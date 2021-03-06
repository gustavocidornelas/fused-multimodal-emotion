"""
Created on Mon April 8, 2019

@author: Gustavo Cid Ornelas
"""
import numpy as np
import random
import tensorflow as tf
import pickle


class ProcessDataText:
    """
    Deals with the text (transcripts) data. Loads all of the text data and labels and splits it into training and
    testing sets

    Attributes
    ----------
    text_data (array): array of shape [num_samples, 128]. Each row corresponds to a transcription
    labels (array): array of shape [num_samples] corresponding to the categories
    dict_size (int): size of the dictionary
    """
    def __init__(self, data_path):

        self.data_path = data_path  # e. g. '../data/processed-data/'

        # loading the audio data and the labels
        self.text_data = self._load_text()
        self.labels = self._load_labels()

        # obtaining the size of the dictionary
        with open(data_path + 'dic.pkl', 'rb') as f:
            self.dict_size = len(pickle.load(f))

    def _load_text(self):
        """
        Loads the transcriptions in the  FC_trans.npy file  and creates an array with all of the transcription samples

        Returns
        ----------
        text_data (array): array of shape [num_samples, 128]. Each row corresponds to a transcription
        """
        print('Loading the transcriptions...')
        text_file = self.data_path + 'FC_trans.npy'

        # reading the file with the transcriptions
        text_data = np.load(text_file)

        return text_data

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
        train_text_data, test_text_data, val_text_data (array): arrays that correspond to the transcriptions
        train_labels, test_labels, val_labels (array): arrays with the labels in the same order as the text_data
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
        data = np.concatenate((self.text_data, self.labels.reshape(num_samples, 1)), axis=1)

        # shuffling the data
        np.random.shuffle(data)

        data = data.astype(int)

        # splitting the data
        train_text_data = data[:num_train + 1, :-1]
        train_labels = data[:num_train + 1, -1]
        test_text_data = data[num_train + 1: num_train + 1 + num_test + 1, :-1]
        test_labels = data[num_train + 1: num_train + 1 + num_test + 1, -1]
        val_text_data = data[num_train + 1 + num_test + 1:, :-1]
        val_labels = data[num_train + 1 + num_test + 1:, -1]

        return train_text_data, train_labels, test_text_data, test_labels, val_text_data, val_labels

    def label_one_hot(self, label, num_categories):
        """
        Converts the labels to the format one-hot, which is expected by our model

        Parameters
        ----------
        label (array): array of integers that specify the category
        num_categories (int): integer that specifies the number of categories

        Returns
        ----------
        one_hot_label (array): array of shape [len(label), num_categories] with a 1 in the corresponding category
        """
        print('Converting the labels to one-hot format...')

        # array that stores all of the one-hot vectors for the labels [samples, num_categories]
        one_hot_label = np.zeros((len(label), num_categories))

        for idx, element in enumerate(label):
            one_hot_label[idx, int(element)] = 1

        return one_hot_label

    def get_glove(self):
        """
        Loads and returns the pre-trained GloVe embedding

        Returns
        ----------
        (array): array of shape [dict_size, 300] with the word embeddings for each word on the dict
        """
        return np.load(self.data_path + 'W_embedding.npy')

    def create_datasets(self, train_text_data, train_labels, test_text_data, test_labels, val_text_data, val_labels,
                        batch_size, num_epochs):
        """
        Creates the training and validation datasets and returns the iterators, next elements of the dataset and the
        handle

        Parameters
        ----------
        train_text_data (array): array of shape [num_train_samples, 128] with training samples as rows
        train_labels (array): array of shape [num_train_samples, num_categories] labels for training
        test_text_data (array): array of shape [num_test_samples, 128] with testing samples as rows
        test_labels (array): array of shape [num_test_samples, num_categories] labels for testing
        val_text_data (array): array of shape [num_val_samples, 128] with validation samples as rows
        val_labels (array): array of shape [num_val_samples, num_categories] labels for validation
        batch_size (int): batch size for the training set
        num_epochs (int): number of epochs

        Returns
        ----------
        train_iterator (Iterator object): iterator for the training dataset
        test_iterator (Iterator object): iterator for the test dataset
        val_iterator (Iterator object): iterator for the validation dataset
        text_input (tensor): tensor with the text inputs to be fed to the model
        label_batch (tensor): tensor with the labels to be fed to the model
        handle (string): handle string, to switch between the datasets
        """
        with tf.name_scope('dataset'):
            # creating the training dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((train_text_data, train_labels))
            train_dataset = train_dataset.repeat(num_epochs)
            train_dataset = train_dataset.batch(batch_size)

            # creating the test dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((test_text_data, test_labels))
            test_dataset = test_dataset.batch(test_text_data.shape[0])

            # creating the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((val_text_data, val_labels))
            val_dataset = val_dataset.batch(val_text_data.shape[0])

            # creating the iterators from the datasets
            train_iterator = train_dataset.make_one_shot_iterator()
            test_iterator = test_dataset.make_initializable_iterator()
            val_iterator = val_dataset.make_initializable_iterator()

            # creating the handle
            handle = tf.placeholder(tf.string, shape=[], name='handle')

            # creating iterator
            iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                           train_dataset.output_shapes)

            # getting the next element
            text_input, label_batch = iterator.get_next()

        return train_iterator, test_iterator, val_iterator, text_input, label_batch, handle
