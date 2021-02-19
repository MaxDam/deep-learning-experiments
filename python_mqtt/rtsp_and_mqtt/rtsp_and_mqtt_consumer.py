#pip install paho-mqtt

import base64
import cv2 as cv
import numpy as np
import paho.mqtt.client as mqtt

RTSP_STREAM = "rtsp://localhost:8554/mystream"
MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC_RECEIVE = "myhome/mx/cserver/detect"

def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))
	client.subscribe(MQTT_TOPIC_RECEIVE)
	
def on_message(client, userdata, msg):
	global frameDetail
	img = base64.b64decode(msg.payload)
	npimg = np.frombuffer(img, dtype=np.uint8)
	frameDetail = cv.imdecode(npimg, 1)
	
frameDetail = np.zeros((240, 320, 3), np.uint8)
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, 1883)
client.loop_start()

cap = cv.VideoCapture(RTSP_STREAM)

while(cap.isOpened()):
	ret, frame = cap.read()
	#frameOut = frame.copy()
	#frame[100:340,100:420,:] = frameDetail[0:240,0:320,:]
	cv.imshow('Frame', frame)
	cv.imshow('FrameDetail', frameDetail)
	if cv.waitKey(20) & 0xFF == ord('q'):
		break
cap.release()
cv.destroyAllWindows()
client.loop_stop()