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

model = Sequential()
model.add(layers.Convolution2D(64, (4, 4), padding='same', input_shape=(conf['height'], conf['width'], 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(64, (2, 2), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Dropout(0.15))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64))
model.add(layers.Activation('tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(conf['num_classes'], activation='softmax'))

model.summary()

model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
datagen = ImageDataGenerator(validation_split=conf['validation_split'],
                             rescale=1. / 255)
train_batches = datagen.flow_from_directory(
    conf['train_path'],
    target_size=(conf['height'], conf['width']),
    subset='training',
    batch_size=conf['batch_size'],
    shuffle=True,
    )
valid_batches = datagen.flow_from_directory(
    conf['train_path'],
    target_size=(conf['height'], conf['width']),
    subset='validation',
    batch_size=conf['batch_size'],
    )

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=200)
mc = ModelCheckpoint('best_model.h5',
                     monitor='val_acc',
                     mode='max',
                     verbose=1,
                     save_best_only=True)
history = model.fit_generator(
    train_batches,
    validation_data=valid_batches,
    epochs=conf['epochs'],
    steps_per_epoch=(file_calculations()[0]*0.8)//conf['batch_size'],
    validation_steps=(file_calculations()[0]*0.2)//conf['batch_size'],
    verbose=1,
    callbacks=[es]
    )
model.save_weights('ownmodel_weights_1.h5')

# Setup plotting of history
fig = plt.figure()
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_acc']

epochs = range(1, len(acc)+1)
plt.subplot(121)
plt.title("Model Accuracy", pad=-40)
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.plot(epochs, acc, 'bo', label="Training Acc")
plt.plot(epochs, val_acc, 'v', label="Validation Acc")
plt.legend()

plt.subplot(122)
plt.title("Cross-Entropy Loss", pad=-40)
plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'v', label="Validation Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.show()
fig.savefig(conf['directory'] + '/ownCNN-accuracy-loss.jpg')
# plt.close(fig)