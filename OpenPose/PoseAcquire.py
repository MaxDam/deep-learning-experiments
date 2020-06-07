import cv2
import time
import numpy as np
import threading

class Pose():

    def __init__(self, mode="COCO", threadCount=2, inWidth = 120, inHeight = 120, threshold = 0.1):

        if mode is "COCO":
            #http://cocodataset.org
            protoFile = "pose/coco/pose_deploy_linevec.prototxt"
            weightsFile = "pose/coco/pose_iter_440000.caffemodel"
            self.nPoints = 18
            self.posePairs = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

        elif mode is "MPI" :
            #http://human-pose.mpi-inf.mpg.de
            protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
            weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
            self.nPoints = 15
            self.posePairs = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

        self.threadCount    = threadCount
        self.inWidth        = inWidth
        self.inHeight       = inHeight
        self.threshold      = threshold
        self.lock           = threading.Lock()
        
        #init neural networks openpose
        self.net = [None] * self.threadCount
        self.inProgress = [None] * self.threadCount
        self.output = None
        for i in range(self.threadCount):
            self.net[i] = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
            self.inProgress[i] = False


    #return last pose and start thread for acquire new pose from current frame
    def getPose(self, frame):
        for i in range(self.threadCount):
            if not self.inProgress[i]:
                thread = threading.Thread(target=self.getPoseAsync, args=(i, frame,))
                thread.start()
                break
        
        output = None
        self.lock.acquire()
        try:
            output = self.output
        finally:
            self.lock.release()
        return output
        

    #return pose from openpose in async
    def getPoseAsync(self, i, frame):
        #startTime = time.time()
        #print("start thread " + str(i))
        self.inProgress[i] = True        
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)
        self.net[i].setInput(inpBlob)
        output = self.net[i].forward()

        #salva output e frames
        self.lock.acquire()
        try:
            self.output = self.getKeyPoints(frame, output)
        finally:
            self.lock.release()

        self.inProgress[i] = False
        #endTime = time.time()
        #print("end thread %s in %s secondi" % (i, (endTime-startTime)))


    #return keypoints and skeleton from output
    def getKeyPoints(self, frame, output):

        #frame dimensions
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
            
        #output dimensions
        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints and skeleton
        keypoints = []
        skeleton = []

        #point iteration
        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > self.threshold : 
                # Add the point to the list if the probability is greater than the threshold
                keypoints.append((int(x), int(y)))
            else :
                keypoints.append(None)

        
        #Build Skeleton
        for pair in self.posePairs:
            partA, partB = pair

            if keypoints[partA] and keypoints[partB]:
                skeleton.append((keypoints[partA], keypoints[partB]))
                
        return (keypoints, skeleton)

