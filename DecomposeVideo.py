import cv2

# Vidcap holds the video from the provided mp4 file.
vidcap = cv2.VideoCapture("Calibration Video/20210525_185120.mp4")
# This function will read the first frame of the mp4 file. In ret, it will hold a boolean file that is true if
# the frame was in ret. img contains the frame itself.
# If vidcap.read() is called again, the next frame is checked.
ret, img = vidcap.read()
count = 0

# This loop continues if readable frames are able to be read from the mp4 file.
while ret:
    cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file as frame(number).jpg
    ret,img = vidcap.read()                     # The next frame is checked and saved in img
    print('Read a new frame: ', ret)            # In the dialog, this will output if a new frame is read properly
    count += 1                                  # Count increased by one to determine how many frames were read.
