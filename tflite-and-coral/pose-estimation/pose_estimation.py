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

#https://github.com/joonb14/TFLitePoseEstimation/blob/main/pose%20estimation.ipynb

import cv2 as cv
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os
from enum import Enum
import math

RTSP_STREAM = "rtsp://admincamera:adminpwd@192.168.1.14:554/stream1"
MODEL_PATH  = "model/posenet_mobilenet_v1_075_721_1281_quant_decoder.tflite"
MODEL_PATH_EDGETPU  = "model/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite"
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
keypointPositions = []
confidenceScores  = []
score = 0
		
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR= 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
	
bodyJoints = np.array([
	(LEFT_WRIST,     LEFT_ELBOW),
	(LEFT_ELBOW,     LEFT_SHOULDER),
	(LEFT_SHOULDER,  RIGHT_SHOULDER),
	(RIGHT_SHOULDER, RIGHT_ELBOW),
	(RIGHT_ELBOW,    RIGHT_WRIST),
	(LEFT_SHOULDER,  LEFT_HIP),
	(LEFT_HIP,       RIGHT_HIP),
	(RIGHT_HIP,      RIGHT_SHOULDER),
	(LEFT_HIP,       LEFT_KNEE),
	(LEFT_KNEE,      LEFT_ANKLE),
	(RIGHT_HIP,      RIGHT_KNEE),
	(RIGHT_KNEE,     RIGHT_ANKLE)
])
  
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
	
def detect(acquiredFrame):
	global keypointPositions, confidenceScores, score
	
	#prepara l'input
	input_frame = cv.resize(acquiredFrame, (1281, 721))
	#input_data = np.array(input_frame).astype('uint8')
	input_data = np.array(input_frame)
	input_data = (input_data/255).astype('float32')
	interpreter.set_tensor(input_details[0]['index'], [input_data])
	
	#effettua la predizione
	interpreter.invoke()
	#print(input_details)
	#print(output_details)
	
	#ottiene i risultati
	
	# 9 * 9 * 17 contains heatmaps
	heatmaps = interpreter.get_tensor(output_details[0]['index'])[0]
	print(heatmaps.shape)
	
	# 9 * 9 * 34 contains offsets
	offsets = interpreter.get_tensor(output_details[1]['index'])[0]
	print(offsets.shape)
	
	# 9 * 9 * 32 contains forward displacements
	forward_displacements = interpreter.get_tensor(output_details[2]['index'])[0]
	print(forward_displacements.shape)
	
	# 9 * 9 * 32 contains backward displacements
	backward_displacements = interpreter.get_tensor(output_details[3]['index'])[0]
	print(backward_displacements.shape)
	
	height = heatmaps.shape[0]
	width = heatmaps[0].shape[0]
	numKeypoints = heatmaps[0][0].shape[0]
	
	#keipoints
	keypointPositions = []
	for keypoint in range(numKeypoints):
		maxVal = heatmaps[0][0][keypoint]
		maxRow = 0
		maxCol = 0
		for row in range(height):
			for col in range(width):
				if (heatmaps[row][col][keypoint] > maxVal):
					maxVal = heatmaps[row][col][keypoint]
					maxRow = row
					maxCol = col
		keypointPositions.append([maxRow,maxCol])
		
	#confidences
	confidenceScores=[]
	yCoords = []
	xCoords = []
	for idx, position in enumerate(keypointPositions):
		positionY = keypointPositions[idx][0]
		positionX = keypointPositions[idx][1]
		yCoords.append(position[0] / (height - 1) * 257 + offsets[positionY][positionX][idx])
		xCoords.append(position[1] / (width - 1) * 257 + offsets[positionY][positionX][idx + numKeypoints])
		confidenceScores.append(sigmoid(heatmaps[positionY][positionX][idx]))
	score = np.average(confidenceScores)

#mostra i collegamenti
def displayJoint(acquiredFrame):
	minConfidence = 0.5

	if (score > minConfidence):
		for line in bodyJoints:
			cv.line(acquiredFrame,(xCoords[line[0].value],yCoords[line[0].value]),(xCoords[line[1].value],yCoords[line[1].value]),(255,0,0),5)
			cv.circle(acquiredFrame, (xCoords[line[0].value], yCoords[line[0].value]), radius=0, color=(0, 0, 255), thickness=-1)
			cv.circle(acquiredFrame, (xCoords[line[1].value], yCoords[line[1].value]), radius=0, color=(0, 0, 255), thickness=-1)
	return acquiredFrame

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
	frame = displayJoint(frame)

	#mostra il frame
	cv.imshow('frame', frame)
	
	#si blocca se viene premuto il tasto 'q'
	if cv.waitKey(20) & 0xFF == ord('q'):
		break

#rilascia gli oggetti		
cap.release()
cv.destroyAllWindows()


