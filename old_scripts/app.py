import tensorflow as tf
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
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


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(conf["height"], conf["width"], 3))

batch_size = conf["batch_size"]
datagen = ImageDataGenerator(rescale=1./255)
# conv_base.trainable = True
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

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
        feature_bat = conv_base.predict(input_bat)
        features[i * batch_size: (i+1) * batch_size] = feature_bat
        labels[i * batch_size: (i+1) * batch_size] = label_bat
        i += 1 
        if i * batch_size >= samples:
            break
    return features, labels
train_features, train_labels = extract_features(conf["train_path"], 2000)
validation_features, validation_labels = extract_features(conf["validation_path"], 1000)
test_features, test_labels = extract_features(conf["test_path"], 1000)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=conf["batch_size"],
                    validation_data=(validation_features, validation_labels),
                    verbose=1)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_acc']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label="Training Acc")
plt.plot(epoch, val_acc, 'v', label="Validation Acc")
plt.title("Training and Validation Acc")

plt.figure()

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epoch, val_loss, 'v', label="Validation Loss")
plt.title("Training and Validation loss")

plt.show()