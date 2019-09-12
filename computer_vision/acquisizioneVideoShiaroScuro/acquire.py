import cv2
import time

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    sleep_time = 0.02
    
    video_capture.set(cv2.CAP_PROP_EXPOSURE, 0)
    time.sleep(sleep_time)
    retChiaro, frameChiaro = video_capture.read()
    
    video_capture.set(cv2.CAP_PROP_EXPOSURE, -8.0)
    time.sleep(sleep_time)
    retScuro, frameScuro = video_capture.read()
    
    cv2.imshow('Video', frameChiaro)
    cv2.imshow('VideoScuro', frameScuro)

    filename = time.strftime("%Y%m%d-%H%M%S")+'.jpg'
    cv2.imwrite("chiaro/"+filename, frameChiaro)
    cv2.imwrite("scuro/"+filename, frameScuro)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()