##################################
# Paths
##################################
data_path = '../data/processed-data/'

##################################
# General
##################################
num_categories = 4

##################################
# Training
##################################
# batch size
batch_size = 16
# learning rate
learning_rate = 3e-4
# number of epochs
num_epochs = 1

##################################
# Audio
##################################
# number of filters in each convolutional layer
num_filters_audio = [25, 50]
# kernel size for each convolutional layer
filter_lengths_audio = [25, 5]
# pooling size (kernel and stride) for each convolutional layer
n_pool_audio = [50, 50]
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
# 'True' if training the multimodal model without attention (use hidden states from the audio model in the text)
multimodal_model_status = False