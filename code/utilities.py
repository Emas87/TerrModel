import cv2 as cv
import os
def deleteRectangle(image, x1, y1, w, h):
    pt1 = (int(x1), int(y1))
    pt2 = (int(x1 + w), int(y1 + h))
    cv.rectangle(image, pt1, pt2, (0,0,0), -1)
    #return image

def deleteHearts():
    dir = os.path.join("positive")
    files = os.listdir(dir)
    images = []
    for file in files:
        # iterate one by one
        images.append(os.path.join('.',dir,file))
    for file in images:
        im_rgb = cv.imread(file)
        x1 = 1589.9
        y1 = 14.35
        w = 299.18
        h = 38.51
        deleteRectangle(im_rgb,x1, y1, w, h)
        x1 = 1589.9
        y1 = 0
        w= 299.18
        h = 40.19
        deleteRectangle(im_rgb,x1, y1, w, h)
        cv.imwrite(file, im_rgb)
        """cv.imshow('Matches', im_rgb)
        while True:
            key = cv.waitKey(1)
            if key == ord('q'):
                cv.destroyAllWindows()
                break"""
        
if __name__ == "__main__":
    deleteHearts()