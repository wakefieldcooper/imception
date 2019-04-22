
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import time
import json
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
fntSize = 0.5
fntThickness = 1
colour = (0,70,255)
def Draw_Text(img, sTxt, aX=30, aY=30):
    if ""==sTxt: return
    cv2.putText(frame, str(sTxt) ,(aX,aY), font, 
        fntSize,(0,255,255), fntThickness,cv2.LINE_AA)
Classes = ['fake', 'real']
with open('config.json') as f:
    conf = json.load(f)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
camera = cv2.VideoCapture(0)
# load the trained convolutional neural network
print("[INFO] loading network...")
json_file = open(conf['FT_model'], 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#load weights into new model
model.load_weights(conf['FT_weights'])
print("loaded model from disk")

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 224 pixels
    return_value, frame = camera.read()  
    imgInfo = np.asarray(frame).shape     
    image = cv2.resize(frame, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

    label = "Real" if prediction == 1 else "Fake"
    label = "{}".format(label)

    # output = cv2.resize(frame, (224, 224))
    # cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7, (0, 255, 0), 2)

    try:    
        Draw_Text(frame, "{}".format(label))
        cv2.namedWindow("Facial Recognition - Entry into Doorway", cv2.WINDOW_NORMAL) 
        cv2.imshow('Facial Recognition - Entry into Doorway', frame)

            
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  #esc   ord('s'):
      
            break
            
    except ValueError:
        break