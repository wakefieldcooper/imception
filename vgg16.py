import tensorflow as tf
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import json

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
with open('config.json') as f:
    conf = json.load(f)

vgg16_model = VGG16(weights='imagenet',
                    input_shape=(conf["height"], conf["width"], 3),
                    include_top=False)
# vgg16_model.layers.pop()


model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
for layer in model.layers[:-7]:
    layer.trainable = False
for layer in model.layers[-7:]:
    layer.trainable = True
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])
datagen = ImageDataGenerator(validation_split=0.2)
train_batches = datagen.flow_from_directory(conf['train_path'], 
                target_size=(224, 224), subset='training',
                batch_size=conf['batch_size'])
valid_batches = datagen.flow_from_directory(conf['train_path'], 
                target_size=(224, 224), subset='validation',
                batch_size=conf['batch_size'])
# test_batches = datagen.flow_from_directory(conf['test_path'], 
#                 target_size=(224, 224),
#                 batch_size=conf['batch_size'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.fit_generator(train_batches, validation_data=valid_batches, 
                    epochs=5, verbose=1)
model.save_weights('imception_weights_2.h5')
model.summary()
