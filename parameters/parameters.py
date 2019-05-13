
##################################
# Paths
##################################
# TODO: add all of the relevant paths here instead of hardcoding them
data_path = '../data/processed-data/'

##################################
# General
##################################
num_categories = 4

##################################
# Training
##################################
# batch size
batch_size = 32
# number of training steps
num_training_steps = 10000
# learning rate
learning_rate = 3e-3
# number of epochs
num_epochs = 5

##################################
# Audio
##################################
# number of filters in each convolutional layer
num_filters_audio = [4, 8]
# kernel size for each convolutional layer
filter_lengths_audio = [25, 5]
# pooling size (kernel and stride) for each convolutional layer
n_pool_audio = [50, 100]
# number of neurons in each layer at the RNN
hidden_dim_audio = 200
# number of layers in a RNN cell
num_layers_audio = 1
# dropout probability (for the RNN)
dr_prob_audio = 0.55

##################################
# Text
################################
# number of neurons in each layer at the RNN
hidden_dim_text = 200
# number of layers in a RNN cell
num_layers_text = 1
# dropout probability (for the RNN)
dr_prob_text = 0.55

##################################
# Multimodal
################################
# 'True' if training the multimodal model (use hidden states from the audio model in the text)
multimodal_model_status = True