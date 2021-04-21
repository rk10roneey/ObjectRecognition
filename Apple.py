#Standard imports
import cv2
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area> 1: #pixel
            cv2.drawContours(imgContour, cnt, -1, (255, 255, 100),3)
            # -1 dilam coz we want contours for all the shapes
            perimeter = cv2.arcLength(cnt,True)
            print(perimeter)
            approx = cv2.approxPolyDP(cnt,0.02*perimeter,True)
            print(approx)
            print(len(approx))
            objectCorner = len(approx)
            x,y,h,w = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+h,y+w),(0,220,50),3) # drawing rectangle for all the shapes
            # img, starting point, ending point, color, border
            #cv2.putText(imgContour,objType,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)


#Read image
image = cv2.imread("F:/apple3.jpg")
image= cv2.resize(image,(680,500))
imgContour = image.copy()
#Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#Color strength parameters in HSV
weaker = np.array([0,0,100])
stronger = np.array([10,255,255])

#Threshold HSV image to obtain input color
mask = cv2.inRange(hsv, weaker, stronger)
#Show original image and result
#cv2.imshow('Image',image)
getContours(mask)
imageArray = ([image, imgContour])
stackedImages = stackImages(0.6, imageArray)
cv2.imshow("WorkFlow", stackedImages)
# cv2.imshow('Result',imgContour)
#Press any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()