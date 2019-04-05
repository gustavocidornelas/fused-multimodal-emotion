
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
num_filters = [4, 8]
# kernel size for each convolutional layer
filter_lengths = [25, 5]
# pooling size (kernel and stride) for each convolutional layer
n_pool = [50, 100]
# number of neurons in each layer at the RNN
hidden_dim = 200
# number of layers in a RNN cell
num_layers = 1
# dropout probability (for the RNN)
dr_prob = 0.55

