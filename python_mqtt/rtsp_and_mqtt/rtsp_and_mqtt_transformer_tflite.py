#pip install tensorflow==2.3.0 --user
#pip install paho-mqtt
import cv2 as cv
import paho.mqtt.client as mqtt
import base64
import numpy as np
import tensorflow as tf
#print(tf.__version__)

RTSP_STREAM = "rtsp://localhost:8554/mystream"
MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC_SEND = "myhome/mx/cserver/detect"
	
DEBUG = True
if DEBUG:
	cap = cv.VideoCapture(0)
else:
	cap = cv.VideoCapture(RTSP_STREAM)
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883)

interpreter = tf.lite.Interpreter(model_path="model/face_detection_front.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
acquire_in_progress = False

def detect_and_send_face(acquiredFrame):
	#(height, width) = acquiredFrame.shape[:2]
	acquiredFrame = cv.resize(acquiredFrame, (128, 128))
	input_data = np.array(acquiredFrame, dtype=np.float32)
	interpreter.set_tensor(input_details[0]['index'], [input_data])
	interpreter.invoke()
	print(output_details)
	rects = interpreter.get_tensor(output_details[0]['index'])
	scores = interpreter.get_tensor(output_details[2]['index'])
	for index, score in enumerate(scores[0]):
		if score > 0.5:
			box = rects[0][index]
			y_min = int(max(1, (box[0] * acquiredFrame.height)))
			x_min = int(max(1, (box[1] * acquiredFrame.width)))
			y_max = int(min(acquiredFrame.height, (box[2] * acquiredFrame.height)))
			x_max = int(min(acquiredFrame.width, (box[3] * acquiredFrame.width)))
			if DEBUG:
				cv.rectangle(acquiredFrame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
				cv.imshow('frame', acquiredFrame)
			else:
				roi_color = acquiredFrame[y_min:y_max, x_min:x_max] 
				send_image(roi_color)
	
def send_image(faceFrame):
	_, buffer = cv.imencode('.jpg', faceFrame)
	jpg_as_text = base64.b64encode(buffer)
	mqtt_client.publish(MQTT_TOPIC_SEND, jpg_as_text)
	
while(cap.isOpened()):
	ret, frame = cap.read()
	
	if not acquire_in_progress:
		detect_and_send_face(frame)
	
	if cv.waitKey(20) & 0xFF == ord('q'):
		break
		
cap.release()
cv.destroyAllWindows()
mqtt_client.disconnect()
