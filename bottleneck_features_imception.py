import tensorflow as tf
from keras.applications import VGG16
from keras import models
from keras.layers import Flatten, Dense, Dropout
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import json
import numpy as np
import math
from datavisualisation import vis_dataset, train_samples, validation_samples

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

with open('config.json') as f:
    conf = json.load(f)

top_model_weights_path = 'fc_model_13gb.h5'
# vis_dataset()


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        conf['train_path'],
        target_size=(conf['height'], conf['width']),
        batch_size=conf['batch_size'],
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, int(math.ceil(train_samples()[0] / conf['batch_size'])),
        verbose=1) #hard code value as can't reference files file_calculations()[0]
    np.save('bottleneck_features_train_13gb.npy',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        conf['validation_path'],
        target_size=(conf['height'], conf['width']),
        batch_size=conf['batch_size'],
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, int(math.ceil(validation_samples()[0] / conf['batch_size'])),
        verbose=1) #file_calculations()[0]
    np.save('bottleneck_features_validation_13gb.npy',
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train_13gb.npy')
    # will need to make validation set to do this method
    train_labels = np.array(
        [0] * (train_samples()[0] // 2) + [1] * (train_samples()[0] // 2))

    validation_data = np.load('bottleneck_features_validation_13gb.npy')
    validation_labels = np.array(
        [0] * (validation_samples()[0] // 2) + [1] * (validation_samples()[0] // 2))

    model = Sequential()
    model.add(Flatten(input_shape=(train_data.shape[1:])))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    model.fit(
        train_data,
        train_labels,
        epochs=conf['epochs'],
        batch_size=conf['batch_size'],
        verbose=1,
        validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
save_bottleneck_features()
train_top_model()
