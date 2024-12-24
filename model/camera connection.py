import cv2
import numpy as np
url = 'https://192.168.0.102:8080/video'
cap = cv2.VideoCapture(url)
while(True):
    
    camera, frame = cap.read()
    resized = cv2.resize(frame,(600,400))
    if frame is not None:
        cv2.imshow("Frame", resized)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

