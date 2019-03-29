
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

##################################
# Audio
##################################
# number of filters in each convolutional layer
num_filters = [4, 8]
filter_lengths = [25, 5]
n_pool = 200
encoder_size = 250
hidden_dim = 200
num_layers = 1

