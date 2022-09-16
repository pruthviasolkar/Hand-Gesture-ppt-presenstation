from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Parameters
width, height = 1280, 720
gestureThreshold = 300
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)                                  
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1) #means if it is 80% sure it is a hand then consider it as hand

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0 # to change the image we will change image number
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image top rigth corner camera view

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len) #sorted based on length so it will not show file 10 after 1
print(pathImages) 

while True:
    # Get image frame
    success, img = cap.read()               
    img = cv2.flip(img, 1) #flip the image of hand because it is moving at opposite side without this(1 means horizontall ,0 means vertical)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber]) # joins the path of two or more files
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10) # line of the screen 

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"] #centre x and centre y
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width])) #xaxis value limitation
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))#yaxis value limitation
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            #gesture 1 -left
            if fingers == [1, 0, 0, 0, 0]: # thumb is the first finger so made first as 1 that means thumb is up
                print("Left")
                buttonPressed = True
                if imgNumber > 0: #if greater then 0 then only it can reduce
                    imgNumber -= 1 #it will reduce the image number so go to the left 
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            #gesture 2 -rigth        
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True #it will wait till the other action
                if imgNumber < len(pathImages) - 1:#if at length -1 till then only add 1
                    imgNumber += 1 #it will increase the image number so go to the right
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            #gesture 3 - pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 300, 0), cv2.FILLED)

        #gesture 4 - draw  
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    if buttonPressed:
        counter += 1 #will press only one time
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall #height(0):hs(defined in variable),width(total width - ws(defined):total width)

    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'): #it will break from the while loop if we press "q " button
        break