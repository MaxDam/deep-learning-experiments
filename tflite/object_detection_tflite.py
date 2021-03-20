#pip install tensorflow==2.3.0 --user
import cv2 as cv
import numpy as np
#from tflite_runtime.interpreter import Interpreter
from tensorflow.lite.python.interpreter import Interpreter
import os
#print(tf.__version__)

RTSP_STREAM = "rtsp://admincamera:adminpwd@192.168.1.7:554/stream1"
	
#cap = cv.VideoCapture(0)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
cap = cv.VideoCapture(RTSP_STREAM, cv.CAP_FFMPEG)
cap.set(cv.CAP_PROP_BUFFERSIZE, 2)

#interpreter = Interpreter(model_path="model/FaceMobileNet_Float32.tflite")
interpreter = Interpreter(model_path="ssd_mobilenet_v1_1_metadata_1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
output_details = interpreter.get_output_details()

busy = False
boxes   = []
classes = []
scores  = []
	
def detect(acquiredFrame):
	#acquiredFrame = cv.resize(acquiredFrame, (112, 112))
	#input_data = np.array(acquiredFrame, dtype=np.float32)
	acquiredFrame = cv.resize(acquiredFrame, (300, 300))
	input_data = np.array(acquiredFrame, dtype=np.uint8)
	interpreter.set_tensor(input_details[0]['index'], [input_data])
	interpreter.invoke()
	#print(output_details)
	boxes   = interpreter.get_tensor(output_details[0]['index'])[0]
	classes = interpreter.get_tensor(output_details[1]['index'])[0]
	scores  = interpreter.get_tensor(output_details[2]['index'])[0]
	#print(scores)
	busy = False
	
while(cap.isOpened()):
	ret, frame = cap.read()
	
	if not busy:
		busy = True
		frame = detect(frame)
		
	for index, score in enumerate(scores):
		if score > 0.5:
			box = boxes[index]
			print(box)
			ymin = int(max(1, (box[0] * height)))
			xmin = int(max(1, (box[1] * width)))
			ymax = int(min(height, (box[2] * height)))
			xmax = int(min(width, (box[3] * width)))
			cv.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
			
	cv.imshow('frame', frame)
	
	if cv.waitKey(20) & 0xFF == ord('q'):
		break
		
cap.release()
cv.destroyAllWindows()
mqtt_client.disconnect()
