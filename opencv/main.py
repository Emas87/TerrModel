import cv2 as cv
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
from matchTemplate import matchImage

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
#wincap = WindowCapture('Terraria')
wincap = WindowCapture('source - Paint')

# load the trained model
cascade_grass = cv.CascadeClassifier('cascade/cascade.xml')
# load an empty Vision class
vision_grass = Vision(None)

images = []
dir = "grass"
files = os.listdir(dir)
for file in files:
    # iterate one by one
    images.append(os.path.join('.',dir,file))

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    if screenshot is None:
        print("ERROR: Window is minimized, please maximize it an rerun the script")
        break

    # do object detection
    rectangles = cascade_grass.detectMultiScale(screenshot)
    #rectangles = matchImage(screenshot, images)

    # draw the detection results onto the original image
    detection_image = vision_grass.draw_rectangles(screenshot, rectangles)

    # display the images
    cv.imshow('Matches', detection_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image, press 'd' to 
    # save as a negative image.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

print('Done.')