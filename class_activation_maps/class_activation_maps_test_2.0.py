from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cv2 import cv2
from skimage import exposure

from glob import glob
import time

def getMap(img):
  img = preprocess_input(np.expand_dims(img, axis=0))
    
  #predict map
  fmaps = model.predict(img)[0] # 7 x 7 x 2048

  #predict class
  probs = resnet.predict(img)
  classnames = decode_predictions(probs)[0]
  #print(classnames)
  probs = np.flip(np.argsort(probs[0]), axis=0)

  camz = np.zeros((5,224,224))
  for i in range(5):
  #for i in range(1):
    pred = probs[i]

    # get the 2048 weights for the relevant class
    w = W[:, pred]

    # "dot" w with fmaps
    cam = fmaps.dot(w)

    # upsample to 224 x 224 (7 x 32 = 224)
    camz[i] = sp.ndimage.zoom(cam, (32, 32), order=1)
  
  return camz, classnames

# add preprocessing layer to the front of VGG
resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
#resnet = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

# make a model to get output before flatten
activation_layer = resnet.get_layer('activation_49')

# create a model object
model = Model(inputs=resnet.input, outputs=activation_layer.output)

# get the feature map weights
final_dense = resnet.get_layer('fc1000')
W = final_dense.get_weights()[0]

#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture("../large_files/NeuralTalk and Walk.mp4")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/4)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080/4)

start_count = 2000
step_count = 10

count = 0
while True:
  ret, frame = capture.read()
  count += 1

  if ret and count > start_count and count % step_count == 0:
    frame = cv2.resize(frame, (224, 224)) 

    camz, classnames = getMap(frame)

    camzIndex = 0
    gray_img = camz[camzIndex]
    gray_img = exposure.rescale_intensity(gray_img, out_range=(0, 255))
    gray_img = np.uint8(gray_img)
    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.5, frame, 0.5, 0)

    #cv2.imshow('frame', frame)
    cv2.imshow('frame', fin)
    print("{} ({:.0f}%)".format(classnames[camzIndex][1],classnames[camzIndex][2]*100))  
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()
