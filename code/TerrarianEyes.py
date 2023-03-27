import cv2 as cv
import numpy as np
import os
import torch
from time import time
from vision import Vision
from windowcapture import WindowCapture
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.plots import Annotator, colors, save_one_box


class TerrarianEyes:

    def __init__(self, tiles_weights_path, objects_weights_path) -> None:
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
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False

        self.loadModels()
        self.loadImages()

    @staticmethod
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
    @staticmethod
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

    def findTiles(self, source):
        #results = self.infereYoloTiles(source)
        #df= results.pandas().xyxy[0]
        #results.show()
        results = self.detectYoloTiles(source)
        img_rgb = source
        final_rectangles = []
        # for each section of tiles look for specific tiles
        for clss, rows in results.items():
            images = self.templates[clss]
            for row in rows:                
                cropped_image = img_rgb[round(row[1]):round(row[1]+row[3]), round(row[0]):round(row[0]+row[2])]
                rectangles = self.matchImage(cropped_image, images)
                #rectangles = matchImageColor(cropped_image, grassImages)

                # convert point from cropped image point to original image point
                for rectangle in rectangles:
                    x1 = int(round(row[0]) + rectangle[0])
                    #x2 = int(round(row.xmin) + rectangle[0] + rectangle[2])
                    y1 = int(round(row[1]) + rectangle[1])
                    #y2 = (round(row.ymin) + rectangle[1] + rectangle[3])
                    #cv.rectangle(img_rgb, (x1, y1), (x2, y2), (0,0,255), 2)
                    final_rectangles.append((x1,y1,rectangle[2],rectangle[3]))

        return final_rectangles

    def findObjects(self, source):
        results = self.detectYoloObjects(source)
        final_rectangles = []
        # for each section of tiles look for specific tiles
        for clss, rows in results.items():
            for row in rows:
                final_rectangles.append(row)

        return final_rectangles

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
        self.tiles_model  = DetectMultiBackend([self.tiles_weights_path], device=self.device, dnn=self.dnn, data=self.data_tiles, fp16=self.half)
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
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
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
                    annotator.box_label(xyxy, label, color=colors(c, True))
        im0 = annotator.result()
        cv2.imshow(str('Matches'), im0)
        cv2.waitKey(1)  # 1 millisecond

        return boxes         

    def infereYoloTiles(self, image):
        #im = image
        im = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        results = self.tiles_model(im)
        return results
      
    def infereYoloObjects(self, image):
        im = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        results = self.objects_model(im)
        return results

    def loadImages(self):
        dir = "grass"
        files = os.listdir(dir)
        grassImages = []
        for file in files:
            # iterate one by one
            grassImages.append(os.path.join('.',dir,file))
        dir = "stone"
        files = os.listdir(dir)
        stoneImages = []
        for file in files:
            # iterate one by one
            stoneImages.append(os.path.join('.',dir,file))
        
        self.templates = {"dirt": grassImages, "stone": stoneImages}

    def startController(self, window_name):
        # initialize the WindowCapture class
        try:
            wincap = WindowCapture(window_name)
        except Exception:
            print("ERROR: Window not found")
            return None

        # load an empty Vision class
        vision_grass = Vision(None)

        loop_time = time()
        while(True):

            # get an updated image of the game
            screenshot = wincap.get_screenshot()
            #screenshot = self.captureWindow()
            if screenshot is None:
                print("ERROR: Window is minimized, please maximize it an rerun the script")
                break

            # do tiles detection
            rectangles_tiles = self.findTiles(screenshot)
            rectangles_objects = self.findObjects(screenshot)
            rectangles = rectangles_objects + rectangles_tiles

            # draw the detection results onto the original image
            detection_image = vision_grass.draw_rectangles(screenshot, rectangles)

            # display the images
            cv.imshow('Matches', detection_image)

            # debug the loop rate
            print('FPS {}'.format(1 / (time() - loop_time)))
            loop_time = time()

            # press 'q' with the output window focused to exit.
            # waits 10 ms every loop to process key presses
            key = cv.waitKey(10)
            if key == ord('q'):
                cv.destroyAllWindows()
                break

        print('Done.')

    def startRecorder():
        # initialize the WindowCapture class
        wincap = WindowCapture('Terraria')

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
            # press 'f' to save screenshot as a positive image, press 'd' to 
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

if __name__ == "__main__":
    # Create Instance
    tiles_weights_path = os.path.join('runs', 'train', 'yolov5l6-tiles', 'weights', 'best.pt')
    objects_weights_path = os.path.join('runs', 'train', 'yolov5l6-objects', 'weights', 'best.pt')
    eyes = TerrarianEyes(tiles_weights_path, objects_weights_path)

    #Inference
    eyes.startController('source - Paint')    

    exit()

    