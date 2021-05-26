#--------------------------------
# Purpose: To understand how OpenCV works and the different functions available in the module
# Date: May 25, 2021
# Author: Bobsy Narayan
#--------------------------------

# Tutorial 1 - OpenCV Python Tutorial #1 - Introduction & Images - Tech with Tim
# Import openCV module
import cv2 as cv
import numpy as np
import random

# Upload images using this function. This function must be used to
# -1, cv.imread_COLOR : Loads a color image. Any transperency in the image will be neglected and is the default.
# 0,  cv.imread_GRAYSCALE : Loads image in grayscale mode
# 1,  cv.imread_UNCHANGED : Loads image as such including alpha channel
img = cv.imread('Calibration images/frame0.jpg', -1)

# This function will resize image to 400pixels by 400pixels.
img = cv.resize(img,(400,400))
# This function will resize image to a multiple of the original function. fx= xsize, fy=ysize
img = cv.resize(img, (0,0), fx=2, fy=2)
# This function will rotate the image.
img: None = cv.rotate(img,cv.cv.ROTATE_180)

# This will show the img. ('name of window output', variable that holds image)
cv.imshow('Image',img)
# This will make the program hold for set time until you click a button
# 0 is infinite time. 5 etc is 5seconds
cv.waitKey(0)
# This will close all windows displayed
cv.destroyAllWindows()

# This will save the image stored in the variable.
cv.imwrite('New_image.jpg',img)


# Tutorial 2 - OpenCV Python Tutorial #2 - Image Fundamentals and Manipulation - Tech with Tim
# When an image is put into python, the computer extracts the pixels from the image and places it in
# a NumPI array.
# This function shows the rows and columns and colours channels in a numpi array.
# Will output in (Height, Width, Channels)
print(img.shape)
# a 2by2 image with 4pixels would like
#[[[0,0,0],[255,255,255]],
#[[0,0,0],[255,255,255]]],
#[blue, green, red]

#img[275][400] give us the specific matrix at row 257 and column 400
#The following loops changes the top 100 rows of the image to be random colours
for i in range(100):
    for j in range(img.shape[1]):
        img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
cv.imshow('Image', img)
# This function is the same as earlier, where infinite time is waited until a user presses a button.
# However, if the s-key is pressed, the if statement will save the image to the computer.
k = cv.waitKey(0)
if k ==ord('s'):
    cv.imwrite("image_copy_2.jpg",img)
cv.destroyAllWindows()

#To copy & paste images
tag=img[0:100, 600:900]
img[100:200, 600:900] = tag

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
