# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import glob
import json
import cv2
from keras.models import model_from_json
np.set_printoptions(suppress=True)

with open('config.json') as f:
    conf = json.load(f)

# load the image
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
 