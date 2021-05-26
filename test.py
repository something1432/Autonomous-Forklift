import cv2
import cv2.aruco as aruco
import numpy as np
import os

import glob

# Function to find chessboard corners for use in camera calibration
def checkerboardCalibration():
    chessboardSize = (8, 6) # not number of squares
    square_size = 0.023

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('Calibration images 2\*.jpg')

    for image in images:

        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, cameraMatrix, cameraDistortion, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    saveCoefficients(cameraMatrix, cameraDistortion)

    return [cameraMatrix, cameraDistortion, rvec, tvec]

def saveCoefficients(mtx, dist):
    cv_file = cv2.FileStorage("calibrationCoefficients.txt", cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


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

# Function to determine distance between 2 aruco markers
def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker
    R, _ = cv2.Rodrigues(rvec2)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec2))
    # invTvec = np.matrix(-tvec2)
    invRvec, _ = cv2.Rodrigues(R)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec

def main():
    cap = cv2.VideoCapture("Tracking target videos/aruco location.mp4")
    frameSize = (1280,720)
    markerLength = 0.020 # size of one side of marker length 45 mmq
    markerTvecList = []
    markerRvecList = []

    # get Camera and distortion matrices
    # cameraMatrix, cameraDistortion, rvec, tvec = checkerboardCalibration()
    cameraMatrix = np.array([[ 1.1792315900622066e+03, 0., 6.6109466233615797e+02], [0., 1.1789853389017303e+03, 3.4401678037318629e+02], [0., 0., 1. ]])
    cameraDistortion = np.array([ 1.6836095183955574e-01, -6.4171147477695833e-01, -2.2543184593057726e-03, 2.7872634899685253e-03, 7.5117009620855502e-01 ])

    # Initialize video recording
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('Tracking.mp4', fourcc, 30.0, (853, 480))

    while True:
        ret, img = cap.read()
        arucoFound = findArucoMarkers(img)

        # Loop through all markers
        if len(arucoFound[0]) != 0:
            del markerTvecList[:]
            del markerRvecList[:]

            for bbox, id in zip(arucoFound[0], arucoFound[1]):
               rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(bbox,markerLength,cameraMatrix,cameraDistortion)
               (rvec - tvec).any() # prevent numpy error?
               aruco.drawAxis(img, cameraMatrix,cameraDistortion, rvec, tvec, 0.01) # Draw Axis

               markerRvecList.append(rvec)
               markerTvecList.append(tvec)
               # print(cameraMatrix)

        # print(markerTvecList)

        # Print distance between two markers
        if len(markerTvecList) == 2:
            composedRvec, composedTvec = relativePosition( markerRvecList[0], markerTvecList[0],  markerRvecList[1], markerTvecList[1])
            distance = np.linalg.norm(composedTvec)
            print(distance)

        # elif len(markerTvecList) == 1:
            # distance = np.linalg.norm(markerTvecList[0])
            # print(distance)

        cv2.imshow("Image", img)

        # Record frame to video
        # resized_img = cv2.resize(img, (853, 480), interpolation=cv2.INTER_AREA)
        # out.write(resized_img)

        # cv2.waitKey(0) == ord('q')

        if cv2.waitKey(24) == ord('q'):
            break


if __name__ == "__main__":
    main()