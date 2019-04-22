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
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys
NAME = "imception-finetune-on-13gb-20epochs-{}".format(int(time.time()))
top_model_weights_path = 'fc_model_13gb.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
with open('config.json') as f:
    conf = json.load(f)

# Build VGG16 base model
model = VGG16(
    weights=conf['weights'],
    input_shape=(conf['width'], conf['height'], 3))  
print('Model loaded.')
# Building full connected classifier

img = image.load_img("random/fake/frame166510.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
class_idx = np.argmax(preds[0])
class_output = model.output[:, class_idx]
last_conv_layer = model.get_layer("block5_conv3")
grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)


img = cv2.imread("random/fake/frame166510.jpg")
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("Original", img)
cv2.imshow("GradCam", superimposed_img)
cv2.waitKey(0)


