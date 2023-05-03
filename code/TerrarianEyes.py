import cv2 as cv
import numpy as np
import os
import torch
import pytesseract
from math import floor
from time import time
from vision import Vision
from windowcapture import WindowCapture
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.augmentations import (letterbox)
from utils.plots import Annotator, colors
from Inventory import Inventory
from Map import Map

TILESZ = 16

class TerrarianEyes:

    def __init__(self, tiles_weights_path, objects_weights_path) -> None:


        # Fast testing

        #
        self.tiles_weights_path = tiles_weights_path
        self.objects_weights_path = objects_weights_path
        self.objects_model = None
        self.tiles_model = None
        self.templates = {}

        # Yolo model
        self.device = select_device('')
        self.dnn = False
        self.data_objects = '../datasets_objects/data.yaml'
        self.data_tiles = '../datasets_tiles/data.yaml'
        self.half = False
        self.imgsz = [640, 640]
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.line_thickness = 8
        self.hide_labels = False
        self.hide_conf = False

        self.loadModels()
        self.loadImages()
        self.inventory = Inventory()
        self.map = Map()

    def matchImage(self, sourceImage, templateImages, threshold = 0.7):
        # Match in gray
        img_gray = cv.cvtColor(sourceImage, cv.COLOR_BGR2GRAY)

        # Apply histogram equalization
        img_gray = cv.equalizeHist(img_gray)

        rectangles = []    

        for image in templateImages:
            template = image
            template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
            template = cv.equalizeHist(template)
            assert template is not None, "image could not be read, check with os.path.exists()"
            w, h = template.shape[::-1]
            try:
                res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
            except cv.error:
                res = []

            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                rectangles.append((pt[0], pt[1], w, h))
        return rectangles

    # Takes twice to finish
    @staticmethod
    def matchImageColor(sourceImage, templateImages):
        rectangles = []    

        for image in templateImages:
            template = cv.imread(image, cv.IMREAD_COLOR)
            assert template is not None, "image could not be read, check with os.path.exists()"
            w, h = template.shape[0:-1]
            res = cv.matchTemplate(sourceImage, template, cv.TM_CCOEFF_NORMED)

            threshold = 0.79
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                rectangles.append((pt[0], pt[1], w, h))
        return rectangles

    def findTiles(self, source):
        results = self.detectYoloTiles(source)
        final_results = {}
        img_rgb = source
        # for each section of tiles look for specific tiles
        for clss, rows in results.items():
            try:
                images = self.templates[clss]
                for row in rows:                
                    cropped_image = img_rgb[round(row[1]):round(row[1]+row[3]), round(row[0]):round(row[0]+row[2])]
                    rectangles = self.matchImage(cropped_image, images)
                    #rectangles = self.matchImageColor(cropped_image, images)

                    # convert point from cropped image point to original image point
                    for rectangle in rectangles:
                        x1 = int(round(row[0]) + rectangle[0])
                        #x2 = int(round(row.xmin) + rectangle[0] + rectangle[2])
                        y1 = int(round(row[1]) + rectangle[1])
                        #y2 = (round(row.ymin) + rectangle[1] + rectangle[3])
                        #cv.rectangle(img_rgb, (x1, y1), (x2, y2), (0,0,255), 2)
                        #final_rectangles.append((x1,y1,rectangle[2],rectangle[3]))
                        if clss not in final_results:
                            final_results[clss] = []
                        final_results[clss].append((x1,y1,rectangle[2],rectangle[3]))
            except KeyError:
                for row in rows:
                    if clss not in final_results:
                        final_results[clss] = []
                    final_results[clss].append((row[0],row[1],row[2], row[3]))

        return final_results
        #return final_rectangles

    def findObjects(self, source):
        results = self.detectYoloObjects(source)
        return results


    @staticmethod
    def showImage(img_rgb):
        # display the images
        cv.imshow('Matches', img_rgb)
        while True:
            key = cv.waitKey(1)
            if key == ord('q'):
                cv.destroyAllWindows()
                break

    def loadModels(self):
        if self.tiles_weights_path is not None:
            self.tiles_model  = DetectMultiBackend([self.tiles_weights_path], device=self.device, dnn=self.dnn, data=self.data_tiles, fp16=self.half)
        if self.objects_weights_path is not None:
            self.objects_model  = DetectMultiBackend([self.objects_weights_path], device=self.device, dnn=self.dnn, data=self.data_objects, fp16=self.half)
    
    def detectYoloTiles(self, source):
        stride, names, pt = self.tiles_model.stride, self.tiles_model.names, self.tiles_model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size

        # Warming inference
        self.tiles_model.warmup(imgsz=(1 if pt or self.tiles_model.triton else bs, 3, *imgsz))  # warmup
        dt = (Profile(), Profile(), Profile())

        #Preprocessing
        im = letterbox(source, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with dt[0]:
            im = torch.from_numpy(im).to(self.tiles_model.device)
            im = im.half() if self.tiles_model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:            
            pred = self.tiles_model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        return self.getBoxes(pred, names, source, im)
        
    def detectYoloObjects(self, source):
        stride, names, pt = self.objects_model.stride, self.objects_model.names, self.objects_model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size

        # Warming inference
        self.objects_model.warmup(imgsz=(1 if pt or self.objects_model.triton else bs, 3, *imgsz))  # warmup
        dt = (Profile(), Profile(), Profile())

        #Preprocessing
        im = letterbox(source, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with dt[0]:
            im = torch.from_numpy(im).to(self.objects_model.device)
            im = im.half() if self.objects_model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:            
            pred = self.objects_model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        
        return self.getBoxes(pred, names, source, im)
    
    def getBoxes(self, pred, names, source, im):
        boxes = {}
        for det in pred:  # per image
            im0 = source.copy()
            #annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # TODO test this part
                """if len(det) > 0:
                    for det1 in det:
                        for *xyxy, conf, cls in reversed(det1):
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            if names[c] not in boxes:
                                boxes[names[c]] = []
                            boxes[names[c]].append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3] - xyxy[1])])
                            # To verify if matchTemplate is curreclty run
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    return boxes"""
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    if names[c] not in boxes:
                        boxes[names[c]] = []
                    boxes[names[c]].append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3] - xyxy[1])])
                    # To verify if matchTemplate is curreclty run
                    #annotator.box_label(xyxy, label, color=colors(c, True))
        #im0 = annotator.result()
        #cv.imshow(str('Matches'), im0)
        #cv.waitKey(1)  # 1 millisecond

        return boxes         

    def loadImages(self):
        dir = "grass"
        files = os.listdir(dir)
        grassImages = []
        for file in files:
            # iterate one by one
            grassImages.append(cv.imread(os.path.join('.',dir,file)))
        dir = "stone"
        files = os.listdir(dir)
        stoneImages = []
        for file in files:
            # iterate one by one
            stoneImages.append(cv.imread(os.path.join('.',dir,file)))
        
        self.templates = {"dirt": grassImages, "stone": stoneImages}
        dir = 'numbers'
        files = os.listdir(dir)
        for file in files:
            # iterate one by one
            name = file.replace('.png','')
            self.templates[name] = [cv.imread(os.path.join('.',dir,file))]

    def startController(self, window_name):
        # initialize the WindowCapture class
        try:
            wincap = WindowCapture(window_name)
        except Exception as e:
            raise Exception() from e


        loop_time = time()
        while(True):
            # get an updated image of the game
            screenshot = wincap.get_screenshot()
            #screenshot = self.captureWindow()
            if screenshot is None:
                print("ERROR: Window is minimized or closed, please maximize it or open it and rerun the script")
                break

            # do tiles detection
            tiles = self.findTiles(screenshot)
            #tiles = {}
            objects = self.findObjects(screenshot)
            self.translateObjects(objects, screenshot)
            self.translateTiles(tiles)
            with open("delete.txt", 'w') as f:
                f.write(str(self.inventory))
                f.write(str(self.map))
            #objects = {}
            annotator = Annotator(screenshot, line_width=int(self.line_thickness/3), font_size = 5, example=str(self.objects_model.names))
            final_rectangles = []
            for clss, rows in tiles.items():
                for row in rows:
                    final_rectangles.append(row)
                    annotator.box_label((row[0], row[1], row[0] + row[2], row[1] + row[3]), clss, color=colors(next((k for k, v in self.tiles_model.names.items() if v == clss), None), True))
            for clss, rows in objects.items():
                for row in rows:
                    final_rectangles.append(row)
                    annotator.box_label((row[0], row[1], row[0] + row[2], row[1] + row[3]), clss, color=colors(next((k for k, v in self.objects_model.names.items() if v == clss), None), True))
            
            #final_rectangles.append([0,0,10,10])
            #self.showImage(screenshot)
            # draw the detection results onto the original image
            #detection_image = vision.draw_rectangles(screenshot, final_rectangles, line_type=self.line_thickness)


            # display the images
            cv.imshow('Matches', screenshot)
            #cv.imshow('Matches', detection_image)

            # debug the loop rate
            print('FPS {}'.format(1 / (time() - loop_time)))
            loop_time = time()

            # press 'q' with the output window focused to exit.
            # waits 10 ms every loop to process key presses
            key = cv.waitKey(10)
            if key == ord('q'):
                cv.destroyAllWindows()
                break
    
    def updateMap(self, screenshot):
        #self.showImage(screenshot)
        # do tiles detection
        self.map = Map()
        tiles = self.findTiles(screenshot)
        self.translateTiles(tiles)
        """annotator = Annotator(screenshot, line_width=int(self.line_thickness/3), font_size = 5, example=str(self.objects_model.names))
        final_rectangles = []
        for clss, rows in tiles.items():
            for row in rows:
                final_rectangles.append(row)
                annotator.box_label((row[0], row[1], row[0] + row[2], row[1] + row[3]), clss, color=colors(next((k for k, v in self.tiles_model.names.items() if v == clss), None), True))
        # display the images
        self.showImage(screenshot)"""

    def updateInventory(self, screenshot):
        # do objects detection
        self.inventory = Inventory()
        objects = self.findObjects(screenshot)
        self.translateObjects(objects, screenshot)

    def startRecorder(self, window_name):
        # initialize the WindowCapture class
        wincap = WindowCapture(window_name)

        loop_time = time()
        while(True):

            # get an updated image of the game
            screenshot = wincap.get_screenshot()
            if screenshot is None:
                print("ERROR: Window is minimized, please maximize it an rerun the script")
                break

            # display the images
            cv.imshow('Matches', screenshot)

            # debug the loop rate
            print('FPS {}'.format(1 / (time() - loop_time)))
            loop_time = time()

            # press 'q' with the output window focused to exit.
            # press 'P' to save screenshot as a positive image, press 'n' to 
            # save as a negative image.
            # waits 1 ms every loop to process key presses
            key = cv.waitKey(1)
            if key == ord('q'):
                cv.destroyAllWindows()
                break
            elif key == ord('p'):
                cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
            elif key == ord('n'):
                cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

    def translateObjects(self, obj_dict, screenshot):
        # rectangles (x,y,w,h)
        # center x + w/2, y + h/2
        for clss, rows in obj_dict.items():
            for obj_row in rows:
                x1 = obj_row[0]
                y1 = obj_row[1]
                w = obj_row[2]
                h = obj_row[3]
                center  = (x1 + w/2, y1 + h/2)
                inventory_min = (20, 20)
                inventory_max = (20 + int(9*52.5) + 50, 20 + int(4*52.5) + 50)
                ammo_min = (587, 115)
                ammo_max = (587 + 34, 115 + int(3*36.5) + 34)
                armor_min = (1817, 473)
                armor_max = (1817 + 50, 473 + int(2*51) + 50)
                build_min = (24, 420)
                build_max = (24 + 54, 690 + int(4*54))
                if center[0] > inventory_min[0] and center[0] < inventory_max[0] and center[1] < inventory_max[1] and center[1] > inventory_min[1]:
                    #Inventory
                    center_diff = (center[0]-inventory_min[0], center[1]-inventory_min[1])
                    slot_size = ((inventory_max[0]-inventory_min[0])/10, (inventory_max[1]-inventory_min[1])/5)
                    col = floor(center_diff[0]/slot_size[0])
                    row = floor(center_diff[1]/slot_size[1])
                    count = self.findNumber(screenshot, center[0] - slot_size[0]/2, center[1], slot_size[0], slot_size[1]/2)
                    self.inventory.updateInventory(row, col, clss, count)                    
                elif center[0] > ammo_min[0] and center[0] < ammo_max[0] and center[1] < ammo_max[1] and center[1] > ammo_min[1]:
                    #Ammo
                    center_diff = center[1]-ammo_min[1]
                    slot_size = (ammo_max[1]-ammo_min[1])/4
                    row = floor(center_diff/slot_size)
                    count = self.findNumber(screenshot, center[0] - slot_size/2, center[1] - slot_size/2, slot_size, slot_size)
                    self.inventory.updateAmmo(row, clss, count)
                    
                elif center[0] > armor_min[0] and center[0] < armor_max[0] and center[1] < armor_max[1] and center[1] > armor_min[1]:
                    #Armor
                    center_diff = center[1]-armor_min[1]
                    slot_size = (armor_max[1]-armor_min[1])/3
                    row = floor(center_diff/slot_size)
                    self.inventory.updateArmor(row, clss, 1)

                elif center[0] > build_min[0] and center[0] < build_max[0] and center[1] < build_max[1] and center[1] > build_min[1]:
                    #Build
                    center_diff = center[1]-build_min[1]
                    slot_size = (build_max[1]-build_min[1])/9
                    row = floor(center_diff/slot_size)
                    self.inventory.updateBuild(row, clss, 1)

                else:
                    #Enemy or object in the groudn, update
                    self.map.updateMap(x1, y1, w, h, clss)

    def translateTiles(self, obj_dict):
        # rectangles (x,y,w,h)
        # center x + w/2, y + h/2
        obj_dict["player"] = [(950, 515, 970 - 950, 560 - 515)]
        for clss, rows in obj_dict.items():
            for obj_row in rows:
                x1 = obj_row[0]
                y1 = obj_row[1]
                w = obj_row[2]
                h = obj_row[3]                
                self.map.updateMap(x1, y1, w, h, clss)

    # delete if not needed anymore
    def calcRec(self, image):
        vision = Vision(None)
        final_rectangles = []
        (20,20)
        # items
        for i in range(0,10):
            for j in range(0,5):
                final_rectangles.append((20 + int(i*52.5), 20 + int(j*52.5), 50, 50))
        (545,115)
        # ammo
        for j in range(0,4):
            final_rectangles.append((587 , 115 + int(j*36.5), 34, 34))
        # armor
        for j in range(0,3):
            final_rectangles.append((1817 , 473 + int(j*51), 50, 50))
        # build
        for j in range(0,4):
            final_rectangles.append((30 , 420 + int(j*54), 40, 40))
        final_rectangles.append((24 , 628, 54, 54))
        for j in range(0,4):
            final_rectangles.append((30 , 690 + int(j*54), 40, 40))
        detection_image = vision.draw_rectangles(image, final_rectangles, line_type=self.line_thickness)
        self.showImage(detection_image)

    def findNumber(self, screenshot, x, y, w, h):
        # crop only 3/4 of image, to avoid any number related to the inventory order
        cropped_image = screenshot[round(y):round(y+h), round(x):round(x+w)]
        numNames = ['0', '1','2','3','4','5','6','7','8','9',]
        numbers = {}
        for numName in numNames:
            images = self.templates[numName]
            rectangles = self.matchImage(cropped_image, images, threshold = 0.45)
            for rectangle in rectangles:
                numbers[int(rectangle[0])] = numName
        number = ''
        for key in sorted(numbers):
            number += str(numbers[key])
        if number == '':
            return 1
        else:
            return int(number)


if __name__ == "__main__":
    # Create Instance
    tiles_weights_path = os.path.join('runs', 'train', 'yolov5l6-tiles', 'weights', 'best.pt')
    objects_weights_path = os.path.join('runs', 'train', 'yolov5l6-objects', 'weights', 'best.pt')
    eyes = TerrarianEyes(tiles_weights_path, objects_weights_path)
    #eyes = TerrarianEyes("", "")
    """images = eyes.templates['dirt']
    img_rgb = cv.imread('positive/trippy.png')
    rectangles = eyes.matchImage(img_rgb, images)
    vision = Vision(None)
    detection_image = vision.draw_rectangles(img_rgb, rectangles, line_type=eyes.line_thickness)
    # display the images
    while True:
        cv.imshow('Matches', detection_image)
        # press 'q' with the output window focused to exit.
        # waits 10 ms every loop to process key presses
        key = cv.waitKey(10)
        if key == ord('q'):
            cv.destroyAllWindows()
            break"""
    
    #Inference
    try:
        #eyes.startController('.*Paint')    
        eyes.startController('Terraria')    
        #eyes.startRecorder(None)    
    except Exception as e:
        raise Exception() from e

    exit()

