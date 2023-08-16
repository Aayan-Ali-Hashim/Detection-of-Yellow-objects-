import numpy as np
import cv2
def detect_yellow_objects(frame):
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])
    yellow_mask = cv2.inRange(hsv_frame,lower_yellow,upper_yellow)
    contours,_ = cv2.findContours(yellow_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x , y , w , h = cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    return frame

#opening webcam
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    if not ret:
        break
    yellow_detected = detect_yellow_objects(frame)
    cv2.imshow('Yellow Object Detection', yellow_detected)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()