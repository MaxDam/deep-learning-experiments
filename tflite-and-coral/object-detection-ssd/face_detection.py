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

import cv2 as cv
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os

RTSP_STREAM = "rtsp://admincamera:adminpwd@192.168.1.14:554/stream1"
MODEL_PATH  = "model/ssd_mobilenet_v2_face_quant_postprocess.tflite"
MODEL_PATH_EDGETPU  = "model/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
DELEGATE_LINUX = 'libedgetpu.so.1'
DELEGATE_MAC = 'libedgetpu.1.dylib'
DELEGATE_WINDOSS = 'edgetpu.dll'

DELEGATE = DELEGATE_MAC
USE_CORAL = False
USE_RTSP = False


if USE_RTSP:
	os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
	cap = cv.VideoCapture(RTSP_STREAM, cv.CAP_FFMPEG)
	cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
else:
	cap = cv.VideoCapture(0)

conf_threshold = 0.5
nms_threshold = 0.4
	
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
scores  = []
boxes   = []
indices = []
numDetections = 0
	
def detect(acquiredFrame):
	global scores, numDetections, boxes, indices, conf_threshold, nms_threshold
	
	#prepara l'input trasformandolo a 8 bit
	input_frame = cv.resize(acquiredFrame, (320, 320))
	input_data = np.array(input_frame).astype('uint8')
	interpreter.set_tensor(input_details[0]['index'], [input_data])
	
	#effettua la predizione
	interpreter.invoke()
	#print(input_details)
	#print(output_details)
	
	#ottiene i risultati
	boxes   = interpreter.get_tensor(output_details[0]['index'])[0]
	classes = interpreter.get_tensor(output_details[1]['index'])[0]
	scores  = interpreter.get_tensor(output_details[2]['index'])[0]
	numDetections = int(interpreter.get_tensor(output_details[3]['index'])[0])
	#print("boxes: ",   boxes)
	#print("classes: ", classes)
	#print("scores: ",  scores)
	#print("numDetections: ",  numDetections)
	
	#trasforma i boxes (xmin, xmax, ymin, ymax -> xmin, ymin, width, height)
	boxes_out = []
	for index, score in enumerate(scores):
		y_min = int(max(1, (boxes[index][0] * frame_height)))
		x_min = int(max(1, (boxes[index][1] * frame_width)))
		y_max = int(min(frame_height, (boxes[index][2] * frame_height)))
		x_max = int(min(frame_width, (boxes[index][3] * frame_width)))
		w = x_max - x_min
		h = y_max - y_min
		boxes_out.append([x_min, y_min, w, h])
	boxes = boxes_out
	
	#chiama la Non-maximum Suppression ottenendo gli indici univoci
	indices = cv.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
	#print("indices: ",  indices)

#disegna il box e la classe predetta	                
def draw_bounding_box(img, label, confidence, x, y, w, h):
	x = round(x)
	y = round(y)
	x_plus_w = round(x + w)
	y_plus_h = round(y + h)
	color = (0,255,255)
	cv.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
	if label != "":
		cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

	#scorre gli indici e per ogni elemento stampa il box
	for i in indices:
		i = i[0]
		x, y, w, h = boxes[i]
		#print(boxes[i])
		confidence = scores[i]
		draw_bounding_box(frame, "", confidence, x, y, w, h)

	#mostra il frame
	cv.imshow('frame', frame)
	
	#si blocca se viene premuto il tasto 'q'
	if cv.waitKey(20) & 0xFF == ord('q'):
		break

#rilascia gli oggetti		
cap.release()
cv.destroyAllWindows()
