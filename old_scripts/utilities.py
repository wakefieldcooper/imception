import os
import numpy as np
import json
import app
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# load the user configs
with open('config.json') as f:
    conf = json.load(f)
batch_size = conf["batch_size"]
datagen = ImageDataGenerator(rescale=1./255)


def extract_features(directory, samples):
    features = np.zeros(shape=(samples, 4, 4, 512))
    labels = np.zeros(shape=(samples))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(conf["height"], conf["width"]),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for input_bat, label_bat in generator:
        feature_bat = app.conv_base.predict(input_bat)
        features[i*batch_size:(i+1)*batch_size] = feature_bat
        labels[i*batch_size:(i+1)*batch_size] = label_bat
        i += 1 
        if i*batch_size >= samples:
            break
    return features, labels
