"""
Created on Mon April 8, 2019

@author: Gustavo Cid Ornelas
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


class EvaluateText:
    """
    Class that contains the methods to evaluate the text model on the test or validation set
    """

    def evaluate_text_model_val(self, sess, model, val_iterator, handle, val_handle, writer_val):
        """
        Performs the evaluation of the model in the validation set

        Parameters
        ----------
        sess: tensorflow session
        model (TextModel object): text model
        val_iterator (Iterator object): iterator for the validation dataset
        handle (string): handle, to switch between the datasets
        val_handle (string): validation dataset handle
        eval_batch_len (int): length of the evaluation batch

        Returns
        ----------
        val_accuracy (float): accuracy on the validation set
        """
        print('Evaluating on the validation set...')

        # initializing the validation iterator with the validation data
        sess.run(val_iterator.initializer)

        # evaluating the model on the validation dataset
        val_accuracy, val_loss, val_summary = sess.run([model.accuracy, model.loss,
                                                           model.summary_op], feed_dict={handle: val_handle})

        writer_val.add_summary(val_summary, global_step=model.global_step.eval())

        print('Validation accuracy: {:.4f}'.format(val_accuracy))

        return val_accuracy

    def evaluate_text_model_test(self, sess, model, test_iterator, handle, test_handle):
        """
        Performs the evaluation of the model in the test set

        Parameters
        ----------
        sess: tensorflow session
        model (TextModel object): text model
        test_iterator (Iterator object): iterator for the test dataset
        handle (string): handle, to switch between the datasets
        test_handle (string): test dataset handle
        eval_batch_len (int): length of the evaluation batch

        Returns
        ----------
        test_accuracy (float): accuracy on the validation set
        """
        print('Evaluating on the test set...')

        # initializing the test iterator with the test data
        sess.run(test_iterator.initializer)

        # evaluating the model on the test dataset
        test_accuracy, test_loss, true_label, prediction = sess.run([model.accuracy,
                                                                        model.loss, model.labels,
                                                                        model.batch_prediction],
                                                                       feed_dict={handle: test_handle})

        print('Test accuracy: {:.4f}'.format(test_accuracy))

        # creating the confusion matrix
        self._build_confusion_matrix(true_label, prediction)

        return test_accuracy

    def _build_confusion_matrix(self, true_label, prediction):
        """
        Creates and displays the confusion matrix for the test set

        Parameters
        ----------
        true_label (array): true labels for the test set
        prediction (array): model's predictions for the test set
        """
        print('Creating the confusion matrix for the test set...')
        # retrieving the predictions (not in the one-hot format)
        true_label = np.argmax(true_label, axis=1)
        prediction = np.argmax(prediction, axis=1)

        # creating the confusion matrix
        cm = confusion_matrix(true_label, prediction)
        # normalizing
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # plotting the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Angry', 'Happy', 'Sad', 'Neutral'], yticklabels=['Angry', 'Happy', 'Sad', 'Neutral'],
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # writing the values in the matrix figure
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        plt.show()

