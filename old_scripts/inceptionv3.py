import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import layers
from keras import backend as K
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


base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(299, 299, 3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers:
    layer.trainable = False

datagen = ImageDataGenerator(validation_split=0.2)
train_batches = datagen.flow_from_directory(conf['train_path'], 
                target_size=(299, 299), subset='training',
                batch_size=conf['batch_size'])
valid_batches = datagen.flow_from_directory(conf['train_path'], 
                target_size=(299, 299), subset='validation',
                batch_size=conf['batch_size'])

model.compile(optimizer=optimizers.RMSprop(), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_batches, validation_data=valid_batches, 
                    epochs=2, verbose=1)

# test_batches = datagen.flow_from_directory(conf['test_path'], 
#                 target_size=(224, 224),
#                 batch_size=conf['batch_size'])

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(train_batches, validation_data=valid_batches, 
                    epochs=10, verbose=1)
model.save_weights('imception_weights_V3.h5')
model.summary()
