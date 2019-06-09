# imception
The exploration of PAD using fine tuned deep convolutional neural networks

## Usage
These are to be run sequentially. Relative paths must be maintained for it to work. All hyperparameters of the project are kept in config.json. 

This was conducted for The University of Queensland's Honours thesis for the degree of Engineering (honours) and thus was not developed for wide adoption. 

A requirements file has not been written as this was never intended to be run by anyone else other than myself. However if you would like to understand the workings of this setup, contact me at: enquiries@cooperwakefield.com

### Bottleneck Features
```
python bottleneck_features_imception.py
```
### Fine Tune Model
```
python finetune_imception.py
```
### Visualise Data Balance
```
python datavisualisation.py
```
### Evaluate Model
```
python evaluate_imception.py
```

## Brief summary
Uses the VGG16 pre-trained network, evaluates convolutional base on the dataset and converts to a numpy array, then feeds these numpy's for train and val into the FC layer. This is for computational efficiency. Creates a .h5 file for the weights of this, which is then fed into the final model for updating or fine tuning. This is done by freezing (weights set and can't be updated) all layers up until the last convolutional block (conv_5 block: layer 15) and adding our own FC layer on top for the classification task. The weights of the bottleneck training is then loaded in to model and then fine tuned on the unfrozen layers. 

