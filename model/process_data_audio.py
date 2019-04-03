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

        return audio_data[:128, :]

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

        return labels[:128]

    def split_train_test(self, alpha):
        """
        Splits the data into training and testing sets in the proportion defined by alpha

        Parameters
        ----------
        alpha (float): number between 0 and 1 that determines the proportion of the data that will be used for training

        Returns
        ----------
        train_audio_data, test_audio_data (array): arrays that correspond to the (truncated) raw audio samples
        train_labels, test_labels (array): arrays with the labels in the same order as the audio_data
        """
        print('Splitting the data ...')

        # setting the seed
        random.seed(31)

        # total number of samples in the dataset
        num_samples = len(self.labels)

        if alpha < 0 or alpha > 1:
            raise ValueError('alpha equals ' + str(alpha) + ' is out of range. It must be between 0 and 1')

        num_train = np.ceil(alpha * num_samples)

        # generating a list with the indexes that correspond to the training data
        training_indexes = random.sample(range(num_samples), int(num_train))
        test_indexes = [i for i in range(num_samples) if i not in training_indexes]

        # splitting into training data
        train_audio_data = self.audio_data[training_indexes, :]
        train_labels = self.labels[training_indexes]

        # splitting into test data
        test_audio_data = self.audio_data[test_indexes, :]
        test_labels = self.labels[test_indexes]

        return train_audio_data, train_labels, test_audio_data, test_labels

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

    def get_batches(self, audio_data, labels, batch_size, num_epochs):
        """
        Creates the dataset and iterator objects using the tensorflow data pipeline. Easily repeats the dataset for the
        number of epochs and separates the batches

        Returns
        ----------
        dataset (Dataset object): represents the data as tensors
        iterator (Iterator object): iterates over the dataset
        """
        print('Creating the batches...')

        # starting the tensorflow data pipeline
        dataset = tf.data.Dataset.from_tensor_slices((audio_data, labels))

        # number of times for the dataset iterated num_epochs times
        dataset = dataset.repeat(num_epochs)

        # creating batches
        dataset = dataset.batch(batch_size)

        # creating the iterator
        iterator = dataset.make_one_shot_iterator()

        return dataset, iterator


if __name__ == '__main__':

    path = '../data/processed-data/'
    data = ProcessDataAudio(path)

    data.split_train_test(alpha=0.9)



