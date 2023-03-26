import cv2 as cv
import numpy as np
import os
import time
import torch
from vision import Vision
import pandas

model = None
grassImages = []
stoneImages = []

def matchImage(sourceImage, templateImages):
    # Match in gray
    img_gray = cv.cvtColor(sourceImage, cv.COLOR_BGR2GRAY)
    rectangles = []    

    for image in templateImages:
        template = cv.imread(image, cv.IMREAD_GRAYSCALE)
        assert template is not None, "image could not be read, check with os.path.exists()"
        w, h = template.shape[::-1]
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            rectangles.append((pt[0], pt[1], w, h))
    return rectangles

# Takes twice to finish
def matchImageColor(sourceImage, templateImages):
    rectangles = []    

    for image in templateImages:
        template = cv.imread(image, cv.IMREAD_COLOR)
        assert template is not None, "image could not be read, check with os.path.exists()"
        w, h = template.shape[0:-1]
        res = cv.matchTemplate(sourceImage, template, cv.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            rectangles.append((pt[0], pt[1], w, h))
    return rectangles

def findTiles(source):
    results = infereYolo(source)
    df= results.pandas().xyxy[0]
    img_rgb = cv.imread(source)
    for _,row in df.iterrows():
        if row['name'] == "dirt":
            images = grassImages
        elif row['name'] == "stone":
            images = stoneImages
            
        cropped_image = img_rgb[round(row.ymin):round(row.ymax), round(row.xmin):round(row.xmax)]
        rectangles = matchImage(cropped_image, images)
        #rectangles = matchImageColor(cropped_image, grassImages)

        # convert point from cropped image point to original image point
        for rectangle in rectangles:
            x1 = round(row.xmin) + rectangle[0]
            x2 = round(row.xmin) + rectangle[0] + rectangle[2]
            y1 = round(row.ymin) + rectangle[1]
            y2 = round(row.ymin) + rectangle[1] + rectangle[3]
            cv.rectangle(img_rgb, (x1, y1), (x2, y2), (0,0,255), 2)

        # display the images
        cv.imshow('Matches', img_rgb)
        while True:
            key = cv.waitKey(1)
            if key == ord('q'):
                cv.destroyAllWindows()
                break


def loadModel(weights_path):
    global model
    model = torch.hub.load('.', 'custom', path=weights_path, source='local')

def infereYolo(image):
    results = model([image])
    return results

def loadImages():
    global grassImages
    global stoneImages
    dir = "../opencv/grass"
    files = os.listdir(dir)
    for file in files:
        # iterate one by one
        grassImages.append(os.path.join('.',dir,file))
    dir = "../opencv/stone"
    files = os.listdir(dir)
    for file in files:
        # iterate one by one
        stoneImages.append(os.path.join('.',dir,file))

if __name__ == "__main__":
    # Load model
    weights_path = os.path.join('runs', 'train', 'yolov5l6', 'weights', 'best.pt')
    loadModel(weights_path)
    loadImages()
    time1 = time.time()
    source = '0.png'
    source = '../opencv/source.png'
    #Inference
    findTiles(source)
    time2 = time.time()

    print("It took: " + str(time2 - time1))
    exit()

    