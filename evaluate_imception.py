# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras import models
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import numpy as np
import argparse
import imutils
import glob
import json
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
np.set_printoptions(suppress=True)

with open('config.json') as f:
    conf = json.load(f)

# load the image
def single_image():
    for file in glob.glob("C:/Users/enqui/AppData/Local/Programs/Python/Python36/Thesis/repo/imception/random/*.jpg"):
        image = cv2.imread(file)
        orig = image.copy()
            
        # pre-process the image for classification
        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network
        print("[INFO] loading network...")
        json_file = open(conf['FT_model'], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        #load weights into new model
        model.load_weights(conf['FT_weights'])
        print("loaded model from disk")

        # classify the input image
        prediction = model.predict(image)
        # build the label
        label = "Real" if prediction == 1 else "Fake"
        label = "{}".format(label)

        # draw the label on the image
        output = imutils.resize(orig, width=224)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)


def load_model():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    json_file = open(conf['FT_model'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(conf['FT_weights'])
    print("loaded model from disk")
    # return model


def generate_scores():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    json_file = open(conf['FT_model'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(conf['FT_weights'])
    print("loaded model from disk")

    generator = ImageDataGenerator()
    test_generator = generator.flow_from_directory(
        conf['test_path'],
        target_size=(conf['height'], conf['width']),
        batch_size=conf['batch_size'],
        shuffle=False,
        class_mode='binary'
        )

    # predict crisp classes for test set
    test_generator.reset()
    predictions = model.predict_generator(test_generator, verbose=1)
    predictions = np.concatenate(predictions, axis=0)
    predictions = predictions.astype(int)
    val_trues = (test_generator.classes)

    cf = confusion_matrix(val_trues, predictions)
    precisions, recall, f1_score, _ = precision_recall_fscore_support(
        val_trues, predictions, average='binary'
        )
    # plt.matshow(cf)
    # plt.title('Confusion Matrix Plot')
    # plt.colorbar()
    # plt.xlabel('Precited')
    # plt.ylabel('Actual')
    # plt.show()
    plt.imshow(cf, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix without Normalization')
    plt.xlabel('Predicted\n  F1 Score: {0:.3f}%'.format(f1_score*100))
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(val_trues)))  # length of classes
    class_labels = ['Fake', 'Real']
    tick_marks
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    # plotting text value inside cells
    thresh = cf.max() / 2.
    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(
            j, i, format(cf[i, j], 'd'),
            horizontalalignment='center',
            color='white' if cf[i, j] > thresh else 'black'
            )
    plt.show()

    print('F1 score: %f' % f1_score)
    print('Recall Score: %f' % recall)
    print('precisions: %f' % precisions) 
# generate_scores()
single_image()

