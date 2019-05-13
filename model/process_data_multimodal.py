"""
Created on Fri May 10, 2019

@author: Gustavo Cid Ornelas
"""
from model.process_data_text import *
from model.process_data_audio import *


class ProcessDataMultimodal:
    """
    Deals with both the audio and textual data in the multimodal setting. Contains methods

    Attributes
    ----------
    text_data (array): array of shape [num_samples, 128]. Each row corresponds to a transcription
    audio_data (array): array of shape [num_samples, 250.000]. Each row corresponds to a raw audio file
    labels (array): array of shape [num_samples] corresponding to the categories
    """
    def __init__(self, data_path, text_data_handler, audio_data_handler):

        # getting the data
        self.text_data = text_data_handler.text_data
        self.audio_data = audio_data_handler.audio_data
        self.labels = text_data_handler.labels

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
        train_audio_data, test_audio_data, val_audio_data (array): arrays that correspond to the raw audio samples
        train_labels, test_labels, val_labels (array): arrays with the labels in the same order as the text_data and
                                                       audio_data
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

        # length of the text and audio samples
        text_len = self.text_data.shape[1]
        audio_len = self.audio_data.shape[1]

        # concatenating data and labels to shuffle
        data = np.concatenate((self.text_data, self.audio_data, self.labels.reshape(num_samples, 1)), axis=1)

        # shuffling the data
        np.random.shuffle(data)

        # splitting the data
        train_text_data = data[:num_train + 1, :text_len]
        train_audio_data = data[:num_train + 1, text_len:-1]
        train_labels = data[:num_train + 1, -1]
        test_text_data = data[num_train + 1: num_train + 1 + num_test + 1, :text_len]
        test_audio_data = data[num_train + 1: num_train + 1 + num_test + 1, text_len:-1]
        test_labels = data[num_train + 1: num_train + 1 + num_test + 1, -1]
        val_text_data = data[num_train + 1 + num_test + 1:, :text_len]
        val_audio_data = data[num_train + 1 + num_test + 1:, text_len:-1]
        val_labels = data[num_train + 1 + num_test + 1:, -1]

        # converting the text back to integer
        train_text_data = train_text_data.astype(int)
        test_text_data = test_text_data.astype(int)
        val_text_data = val_text_data.astype(int)

        return train_text_data, train_audio_data, train_labels, test_text_data, test_audio_data, test_labels, \
               val_text_data, val_audio_data, val_labels
