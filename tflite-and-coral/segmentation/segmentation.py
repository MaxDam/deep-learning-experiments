#conda create --name py36 python=3.6
#pip install tensorflow==2.3.0 --user
#pip install tflite==2.3.0
#pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
#conda install -c conda-forge opencv
#conda install -c conda-forge six
#conda install -c conda-forge absl-py
#conda install -c conda-forge google-cloud-bigquery
#conda install -c conda-forge opt_einsum
#conda install -c conda-forge termcolor
#source activate py36

#https://github.com/joonb14/TFLiteSegmentation/blob/main/DeepLabv3.ipynb

import cv2 as cv
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os
from PIL import Image

RTSP_STREAM = "rtsp://admincamera:adminpwd@192.168.1.14:554/stream1"
MODEL_PATH  = "model/keras_post_training_unet_mv2_256_quant.tflite"
MODEL_PATH_EDGETPU  = "model/keras_post_training_unet_mv2_256_quant_edgetpu.tflite"
DELEGATE_LINUX = 'libedgetpu.so.1'
DELEGATE_MAC = 'libedgetpu.1.dylib'
DELEGATE_WINDOSS = 'edgetpu.dll'

DELEGATE = DELEGATE_MAC
USE_CORAL = False
USE_RTSP = False
USE_VIDEO_DEMO = False


if USE_VIDEO_DEMO:
	cap = cv.VideoCapture("video/Pose.mp4")
elif USE_RTSP:
	os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
	cap = cv.VideoCapture(RTSP_STREAM, cv.CAP_FFMPEG)
	cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
else:
	cap = cv.VideoCapture(0)
	
if USE_CORAL:
	interpreter = tflite.Interpreter(model_path=MODEL_PATH_EDGETPU, experimental_delegates=[tflite.load_delegate(DELEGATE)])
else:
	interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
output_details = interpreter.get_output_details()

frame_width  = 0
frame_height = 0
busy = False
output_data = []

labelsArrays = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
      "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
      "person", "potted plant", "sheep", "sofa", "train", "tv"]
	  
def detect(acquiredFrame):
	global output_data
	
	#prepara l'input
	input_frame = cv.resize(acquiredFrame, (256, 256))
	#input_data = np.array(input_frame).astype('uint8')
	input_data = np.array(input_frame)
	input_data = (input_data/255).astype('float32')
	interpreter.set_tensor(input_details[0]['index'], [input_data])
	
	#effettua la predizione
	interpreter.invoke()
	#print(input_details)
	#print(output_details)
	
	#ottiene i risultati
	output_data = interpreter.get_tensor(output_details[0]['index'])[0]

#mostra l'output
def displayOut(acquiredFrame):
	global output_data, labelsArrays
	mSegmentBits = np.zeros((257,257)).astype(int)
	outputbitmap = np.zeros((257,257)).astype(int)
	for y in range(257):
		for x in range(257):
			maxval = 0
			mSegmentBits[x][y]=0
			
			for c in range(21):
				value = output_data[y][x][c]
				if c == 0 or value > maxVal:
					maxVal = value
					mSegmentBits[y][x] = c
					#print(mSegmentBits[x][y])
			#label = labelsArrays[mSegmentBits[x][y]]
			#print(label)
			if(mSegmentBits[y][x]==15):
				outputbitmap[y][x]=1
			else:
				outputbitmap[y][x]=0
				
	temp_outputbitmap= outputbitmap*255
	PIL_image = Image.fromarray(np.uint8(temp_outputbitmap)).convert('L')
	org_mask_img = PIL_image.resize(im.size)
	return org_mask_img

#scorre i frames acquisiti dalla camera
while(cap.isOpened()):
	#ottiene il singolo frame e le sue dimensioni
	ret, frame = cap.read()
	if not ret: continue
	frame_width  = frame.shape[1]
	frame_height = frame.shape[0]

	#effettua la predizione
	if not busy:
		busy = True
		detect(frame)
		busy = False

	#stampa sul frame i punti del corpo trovati
	frame = displayOut(frame)

	#mostra il frame
	cv.imshow('frame', frame)
	
	#si blocca se viene premuto il tasto 'q'
	if cv.waitKey(20) & 0xFF == ord('q'):
		break

#rilascia gli oggetti		
cap.release()
cv.destroyAllWindows()


