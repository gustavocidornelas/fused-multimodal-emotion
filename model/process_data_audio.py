"""
Created on Fri March 22, 2019

@author: Gustavo Cid Ornelas
"""
import numpy as np
import random


class ProcessDataAudio:
    """
    Deals with the audio data. Loads all of the audio data and labels and splits it into training and testing sets

    Attributes
    ----------
    audio_data (list): list of arrays that correspond to the raw audio samples
    labels (list): list with the labels in the same order as the audio_data
    """

    def __init__(self, data_path):

        self.data_path = data_path  # e. g. '../data/processed-data/'

        # loading the audio data and the labels
        self.audio_data = self.load_audio()
        self.labels = self.load_labels()

    def load_audio(self):
        """
        Loads the audio FC_raw_audio.csv file  and creates a list with all the audio samples as arrays

        Returns
        ----------
        audio_data (list): list of arrays that correspond to the raw audio samples
        """
        audio_file = self.data_path + 'FC_raw_audio.csv'

        # reading the audio file
        with open(audio_file) as f:
            lis = [line.split() for line in f]

        # audio samples in the second column
        audio_data = lis[1::2]

        return audio_data

    def load_labels(self):
        """
        Loads the audio FC_label.txt file  and creates a list with all of the labels

        Returns
        ----------
        labels (list): list with the labels in the same order as the audio_data
        """

        labels_file = self.data_path + 'FC_label.txt'

        # reading the labels file
        with open(labels_file) as f:
            labels = f.readlines()

        return labels

    def split_train_test(self, alpha):
        """
        Splits the data into training and testing sets in the proportion defined by alpha

        Parameters
        ----------
        alpha (float): number between 0 and 1 that determines the proportion of the data that will be used for training

        Returns
        ----------
        train_audio_data, test_audio_data (list): list of arrays that correspond to the raw audio samples
        train_labels, test_labels (list): list with the labels in the same order as the audio_data
        """

        # setting the seed
        random.seed(31)

        # total number of samples in the dataset
        num_samples = len(self.labels)

        if alpha < 0 or alpha > 1:
            raise ValueError('alpha equals ' + str(alpha) + ' is out of range. It must be between 0 and 1')

        num_train = np.ceil(alpha * num_samples)

        training_indexes = random.sample(range(num_samples), int(num_train))

        # splitting into training data
        train_audio_data = [self.audio_data[i] for i in training_indexes]
        train_labels = [self.labels[i] for i in training_indexes]

        # splitting into test data
        test_audio_data = [self.audio_data[i] for i in range(num_samples) if i not in training_indexes]
        test_labels = [self.labels[i] for i in range(num_samples) if i not in training_indexes]

        return train_audio_data, train_labels, test_audio_data, test_labels


if __name__ == '__main__':

    path = '../data/processed-data/'
    data = ProcessDataAudio(path)

    data.split_train_test(alpha=0.9)


