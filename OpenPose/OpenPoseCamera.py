import cv2
import time
import numpy as np
import threading

MODE = "COCO"

if MODE is "COCO":
    #http://cocodataset.org
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    #http://human-pose.mpi-inf.mpg.de
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

DRAW_POINT_NUMBER = False
DRAW_SKELETON = True

#inWidth = 368
#inHeight = 368
inWidth = 120
inHeight = 120
threshold = 0.1

cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#get async pose
in_progress = False
output = None
def getPose(frame):
    global output, in_progress, net, inWidth, inHeight
    in_progress = True        
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    in_progress = False

#infinite loop
while cv2.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    #frame dimensions
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    #get async output from openPose
    if not in_progress:
        thread = threading.Thread(target=getPose, args=(frame,))
        thread.start()

    #if output is none continue
    if output is None:
        cv2.imshow('Output', frame)
        continue

    #output dimensions
    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    #point iteration
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
            if DRAW_POINT_NUMBER:
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    
    # Draw Skeleton
    if DRAW_SKELETON:
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    
    cv2.imshow('Output', frame)

