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
        self.text_data = text_data_handler.text_data[:200, :]
        self.audio_data = audio_data_handler.audio_data[:200, :]
        self.labels = text_data_handler.labels[:200]

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

    def create_datasets(self, text_placeholder, audio_placeholder, label_placeholder, test_text_data, test_audio_data, test_labels, val_text_data, val_audio_data,
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
            # TODO: update method description
            # creating the training dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((text_placeholder, audio_placeholder, label_placeholder))
            train_dataset = train_dataset.repeat(num_epochs)
            train_dataset = train_dataset.batch(batch_size)

            # creating the test dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((test_text_data, test_audio_data, test_labels))
            test_dataset = test_dataset.batch(test_labels.shape[0])

            # creating the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((val_text_data, val_audio_data, val_labels))
            val_dataset = val_dataset.batch(val_labels.shape[0])

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
            text_input, audio_input, label_batch = iterator.get_next()

        return train_iterator, test_iterator, val_iterator, text_input, audio_input, label_batch, handle