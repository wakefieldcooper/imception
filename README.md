# imception
The exploration of PAD using deep learning

## Usage
### Bottleneck Features
```
python bottleneck_features.py
```
### Fine Tune Model
```
python vgg16.py
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
Uses the VGG16 pre-trained network currently, evaluates convolutional base on the dataset and numpyfies it (change wording), then feeds these numpy's for train and val into the FC layer to determine where the bottlenecks are in the training. Creates a .h5 file for the weights of this, which is then fed into the final model for updating or fine tuning. This is done by freezing (weight set and can't be updated) all layers up until the last convolutional block (conv_5 block: layer 15) and adding our own FC layer on top for the classification task. The weights of the bottleneck training is then loaded in to model and then fine tuned on the unfrozen layers. 
## To Do 
- Write requirements.txt 
- Establish Parameters on what it was trained on: TF, keras etc. versions
