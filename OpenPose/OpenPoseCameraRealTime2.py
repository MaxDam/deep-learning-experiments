import cv2
import time
import numpy as np
from PoseAcquire2 import Pose

DRAW_POINT_NUMBER = True
DRAW_SKELETON = True

pose = Pose(mode="COCO", threadCount=2, threshold = 0.1)

cap = cv2.VideoCapture(0) #from camera
#cap = cv2.VideoCapture("sample_video.mp4") #from video
hasFrame, frame = cap.read()

#infinite loop
while cv2.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    #get output from openPose
    output = pose.getPose(frame)
    if output is None: continue
    keypoints, skeleton = output
    
    #point iteration
    for i in range(len(keypoints)):
        
        keypoint = keypoints[i]
        if(keypoint is None): continue
        
        cv2.circle(frame, (keypoint[0], keypoint[1]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
        if DRAW_POINT_NUMBER:
            cv2.putText(frame, "{}".format(i), (keypoint[0], keypoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    
    # Draw Skeleton
    if DRAW_SKELETON:
        for i in range(len(skeleton)): 
            pair = skeleton[i]         
            cv2.line(frame, pair[0], pair[1], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, pair[0], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, pair[1], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    
    cv2.imshow('Output', frame)

    time.sleep(0.1)