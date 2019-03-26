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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

with open('config.json') as f:
    conf = json.load(f)

vgg16_model = VGG16(weights=conf['weights'],
                    input_shape=(conf["height"], conf["width"], 3),
                    include_top=conf['include_top'])

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
model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
datagen = ImageDataGenerator(validation_split=conf['validation_split'])
train_batches = datagen.flow_from_directory(conf['train_path'],
                                            target_size=(conf['height'], conf['width']),
                                            subset='training',
                                            batch_size=conf['batch_size'])
valid_batches = datagen.flow_from_directory(conf['train_path'],
                                            target_size=(conf['height'], conf['width']),
                                            subset='validation',
                                            batch_size=conf['batch_size'])
# test_batches = datagen.flow_from_directory(conf['test_path'],
#                                            target_size=(224, 224),
#                                            batch_size=conf['batch_size'])
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=200)
mc = ModelCheckpoint('best_model.h5',
                     monitor='val_acc',
                     mode='max',
                     verbose=1,
                     save_best_only=True)
history = model.fit_generator(train_batches, validation_data=valid_batches,
                              epochs=conf['epochs'],
                              verbose=1,
                              callbacks=[es])
model.save_weights('imception_weights_2.h5')

# Setup plotting of history
fig = plt.figure()
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_acc']

epochs = range(1, len(acc)+1)
plt.subplot(211)
plt.title("Accuracy", pad=-40)
plt.plot(epochs, acc, 'bo', label="Training Acc")
plt.plot(epoch, val_acc, 'v', label="Validation Acc")
plt.legend()

plt.subplot(212)
plt.title("Cross-Entropy Loss", pad=-40)
plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epoch, val_loss, 'v', label="Validation Loss")
plt.legend()

plt.show()
fig.savefig('/accuracy-loss.jpg')
plt.close(fig)