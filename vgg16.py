import tensorflow as tf
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import json
from datavisualisation import vis_dataset, train_samples, validation_samples
import time

NAME = "imception-finetune-on-13gb-20epochs-{}".format(int(time.time()))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

top_model_weights_path = 'fc_model_13gb.h5'


tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

with open('config.json') as f:
    conf = json.load(f)

# Build VGG16 base model
VGG_model = VGG16(
    weights=conf['weights'],
    input_shape=(conf["height"], conf["width"], 3),
    include_top=conf['include_top'])  
print('Model loaded.')

# Building full connected classifier
top_model = Sequential()
top_model.add(layers.Flatten(input_shape=VGG_model.output_shape[1:]))
top_model.add(layers.Dense(256, activation='relu')) 
top_model.add(layers.Dropout(0.5))
top_model.add(layers.Dense(1, activation='sigmoid'))

# Load in weights from bottleneck training.
# Needed in order to conduct fine tuning
print("[INFO] - Loading top model weights")
top_model.load_weights(top_model_weights_path)

# Add FC classifier model to base model
print("[INFO] - Adding top layer")
# model.add(top_model)
model = Model(input=VGG_model.input, output=top_model(VGG_model.output))


# Set up to the last Conv block as non-trainable
# This will preserve weights in these layers
for layer in model.layers[:15]:
    layer.trainable = False

print("[INFO] - Compiling...")
model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Won't apply data augmentation for now
datagen = ImageDataGenerator(rescale=1. / 255)
train_batches = datagen.flow_from_directory(
    conf['train_path'],
    target_size=(conf['height'], conf['width']),
    batch_size=conf['batch_size'],
    class_mode='binary'
    )
valid_batches = datagen.flow_from_directory(
    conf['validation_path'],
    target_size=(conf['height'], conf['width']),
    batch_size=conf['batch_size'],
    class_mode='binary'
    )

es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   verbose=1,
                   patience=7)
mc = ModelCheckpoint('best_model_13gb.h5',
                     monitor='val_acc',
                     mode='max',
                     verbose=1,
                     save_best_only=True)
history = model.fit_generator(
    train_batches,
    validation_data=valid_batches,
    epochs=conf['epochs'],
    steps_per_epoch=(train_samples()[0]//conf['batch_size']),
    validation_steps=(validation_samples()[0]//conf['batch_size']),
    verbose=1,
    callbacks=[tensorboard]
    )
# serialize model to JSON
model_json = model.to_json()
with open("finetuned_vgg16_13gb.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('imception_finetune_13gb.h5')
print("[INFO] - Saved model to disk")