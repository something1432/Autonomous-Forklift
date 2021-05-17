import cv2
import cv2.aruco as aruco
import numpy as np
import os

import glob

# TEST MESSAGE 3
# Function to find chessboard corners for use in camera calibration
def findChessboardCorners():
    chessboardSize = (7, 7)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('Calibration images\*.jpg')

    for image in images:

        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    return [objpoints, imgpoints]


# Function to identify and draw bounding boxes around aruco markers
def findArucoMarkers(img, markerSize=4, totalMarkers=100, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs, ids]

# def augmentAruco(bbox, id, img, imgAug, drawID=True):

def main():
    cap = cv2.VideoCapture("Tracking target videos/aruco cube 3.mp4")
    frameSize = (1280,720)
    markerLength = 0.040 # size of one side of marker length 45 mmq
    objpoints, imgpoints = findChessboardCorners()
    ret, cameraMatrix, cameraDistortion, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('Tracking.mp4', fourcc, 30.0, (853, 480))

    while True:
        ret, img = cap.read()
        arucoFound = findArucoMarkers(img)

        # Loop through all markers
        if len(arucoFound[0]) != 0:
           for bbox, id in zip(arucoFound[0], arucoFound[1]):
               rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(bbox,markerLength,cameraMatrix,cameraDistortion)
               (rvec - tvec).any() # prevent numpy error?
               aruco.drawAxis(img, cameraMatrix,cameraDistortion, rvec, tvec, 0.01) # Draw Axis
               print(tvec)

        cv2.imshow("Image", img)
        # resized_img = cv2.resize(img, (853, 480), interpolation=cv2.INTER_AREA)
        # out.write(resized_img)
        cv2.waitKey(0) == ord('q')

        # if cv2.waitKey(0) == ord('q'):
            # break


if __name__ == "__main__":
    main()