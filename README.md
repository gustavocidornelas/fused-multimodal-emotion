# fused-multimodal-emotion

## Organization

The project is comprised of the folders _data_, _model_, _parameters_ and _preprocessing_. After training a model, two other  folders are created, namely _graphs_ and _pretrained-models_.

### _data_
Contains all the data used, in its raw and preprocessed stages. The dataset used is the [IEMOCAP database] (https://sail.usc.edu/iemocap/).The IEMOCAP is available uppon request in the link provided. The data actually used by our model is preprocessed from the raw dataset. Our preprocessed data can be made available uppon request, provided that the person already has access to the IEMOCAP dataset. 

### _model_
Contains all of the implemented models. All the models have three main files: `train_[model].py`, `process_[model]_data.py`, `evaluate_[model].py`, where `[model]` is the model of interest (`text`, `audio`, `multimodal`, `multimodal_attention`). 

`train_[model].py` contains the main training loop. `process_[model]_data.py` handles the data of the model of interest, loading, batching and splitting the data into train, validation and test sets. `evaluate_[model].py` evaluates the performance of the model in the validation and test sets and creates the confusion matrix for the test set.

### _parameters_
The file `parameters.py` contains all of the relevant parameters for the simulation, divided in sections, according to the model.

### _preprocessing_
The file `prepare_raw_audio.py` reads all the raw audio files that are used by our model, truncates, zero pads and saves them to the expected directory within the _data_ folder. 

### _graphs_
Created when the model starts training. Contains two folders: *graph_train* and *graph_val*, with the information that can be visualized on TensorBoard, including the model's graph and the training and validation accuracies and losses.

### _pretrained-models_
Created once the model is trained. Saves the whole model with its weights, that can be used to do inference later.
