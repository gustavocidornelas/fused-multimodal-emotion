# fused-multimodal-emotion

## Organization

The project is comprised of the folders _data_, _model_, _parameters_ and _preprocessing_. After training a model, two other  folders are created, namely _graphs_ and _pretrained-models_.
###### _data_
Contains all the data used, in its raw and preprocessed stages. The dataset used is the [IEMOCAP dataset](https://sail.usc.edu/iemocap/index.html). The IEMOCAP is available uppon request in the link provided. The data actually used by our model is preprocessed from the raw dataset. Our preprocessed data can be made available uppon request, provided that the person already has access to the IEMOCAP dataset. 
###### _model_
Contains all of the implemented models. All the models have three main files: `train_[model].py`, `process_[model]_data.py`, `evaluate_[model].py`, where `[model]` is the model of interest (`text`, `audio`, `multimodal`, `multimodal_attention`). 
`train_[model].py` contains the main training loop. `process_[model]_data.py` handles the data of the model of interest, loading, batching and splitting the data into train, validation and test sets. `evaluate_[model].py` evaluates the performance of the model in the validation and test sets and creates the confusion matrix for the test set.
###### _parameters_
The file `parameters.py` contains all of the relevant parameters for the simulation, divided in sections, according to the model.
###### _preprocessing_
The file `prepare_raw_audio.py` reads all the raw audio files that are used by our model, truncates, zero pads and saves them to the expected directory within the _data_ folder. 
###### _graphs_
Created when the model starts training. Contains two folders: *graph_train* and *graph_val*, with the information that can be visualized on TensorBoard, including the model's graph and the training and validation accuracies and losses.
###### _pretrained-models_
Created once the model is trained. Saves the whole model with its weights, that can be used to do inference at a later stage.

## Running

To train a model, the first step is obtaining all the data. If you already have all the preprocessed data in the correct folder within the _data_ directory, you are good to go. If you would like to truncate the raw audio files differently, you can edit that in `/preprocessing/prepare_raw_audio.py` and run it once. The preprocessed raw audio files will be saved to the correct directory.

Once you have all the data correctly placed, you can edit the model's parameters in `/parameters/parameters.py`. The parameters for all the models are in this single file, but they are organized in sections, so it is important that you edit the parameters in the correct section.

With the correct parameters, it is time to train. The proportion of the data used to build the training, validation and test sets is hardcoded in `train_[model].py`. To train a model, you should run `train_[model].py`, where `[model]` is one of the options `text`, `audio`, `multimodal`, `multimodal_attention`. The model is evaluated on the validation set every 50 batches, but that can be changed in `train_[model].py` inside the training loop.

After training, the model is evaluated on the test set and the full model is saved to `/pretrained-models`.

## References
The full report with the description of our motivation and approach is avaliable [here]. The code in this repository was developed as part of my semester project at the [Chair for Mathematical Information Science](https://www.mins.ee.ethz.ch/index.html), at [ETH Zürich](https://www.ethz.ch/en.html).

A significant part of the code for this project is built over the code from [multimodal-speech-emotion](https://github.com/david-yoon/multimodal-speech-emotion). 




