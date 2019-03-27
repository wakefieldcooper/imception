import tensorflow as tf
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import json
from datavisualisation import vis_dataset, file_calculations

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

with open('config.json') as f:
    conf = json.load(f)

top_model_weights_path = 'fc_model.h5'
# vis_dataset()


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 validation_split=conf['validation_split'])

    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        conf['train_path'],
        target_size=(conf['height'], conf['width']),
        batch_size=conf['batch_size'],
        subset='training',
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, (file_calculations()[0] // conf['batch_size'])*0.8,
        verbose=1)
    np.save('bottleneck_features_train.npy',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        conf['train_path'],
        target_size=(conf['height'], conf['width']),
        batch_size=conf['batch_size'],
        subset='validation',
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, (file_calculations()[0] // conf['batch_size'])*0.2,
        verbose=1)
    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')

    validation_data = np.load('bottleneck_features_validation.npy')

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )

    model.fit_generator(
        train_data,
        epochs=conf['epochs'],
        batch_size=conf['batch_size'],
        verbose=1,
        validation_data=validation_data)
    model.save_weights(top_model_weights_path)
save_bottlebeck_features()
train_top_model()