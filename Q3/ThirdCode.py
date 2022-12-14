import numpy as np
import cv2



def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr





get = cv2.VideoCapture('Derrick.mov')

fun, prev = get.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

n = 10 
count = 0
while fun:

    fun, img = get.read()
    if(fun and count%n == 0):
        print(count)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray, flow))
    
    count += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break



get.release()
cv2.destroyAllWindows(