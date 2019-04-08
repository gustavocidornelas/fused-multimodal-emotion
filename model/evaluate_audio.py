"""
Created on Mon April 8, 2019

@author: Gustavo Cid Ornelas
"""


class EvaluateAudio:
    """
    Class that contains the methods to evaluate the text model on the test or validation set
    """

    def evaluate_audio_model(self, sess, model, val_iterator, handle, val_handle, writer_val):
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
        _, val_accuracy, val_loss, val_summary = sess.run([model.optimizer, model.accuracy, model.loss,
                                                           model.summary_op], feed_dict={handle: val_handle})

        writer_val.add_summary(val_summary, global_step=model.global_step.eval())

        print('Validation accuracy: {:.4f}'.format(val_accuracy))

        return val_accuracy



