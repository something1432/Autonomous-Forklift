import cv2

vidcap = cv2.VideoCapture("Calibration Video/20210525_185120.mp4")
ret, img = vidcap.read()
count = 0

while ret:
    cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file
    ret,img = vidcap.read()
    print('Read a new frame: ', ret)
    count += 1
