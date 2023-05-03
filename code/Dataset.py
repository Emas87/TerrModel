import argparse
import os
import shutil
import cv2 as cv
from utilities import deleteRectangle
from TerrarianEyes import TerrarianEyes
import random
# Code that reads a dataset
class Dataset:
    def __init__(self, dir) -> None:
        self.origin = dir
        self.labels = {}

        # Get classes
        self.classes = []
        with open(os.path.join(dir, "data.yaml"), "r") as f:
            lines = f.readlines()
        for line in lines:
            if "names: " in line:
                classesStr = line.replace("names: ",'').replace('\'', '').replace('[','').replace(']', '').replace('\n', '')
                self.classes = classesStr.split(', ')
        
        # Get labels
        label_files = os.listdir(os.path.join(dir, 'train', 'labels'))
        for label_file in label_files:
            with open(os.path.join(dir, 'train', 'labels', label_file), "r") as f:
                lines = f.readlines()
            file_name = os.path.splitext(label_file)[0]
            self.labels[file_name] = []
            for line in lines:
                self.labels[file_name].append(line.split(' '))
                
    # deletes all matches
    def deleteMatches(self):
        if os.path.exists("images_tmp_delete"):
            shutil.rmtree("images_tmp_delete")
        shutil.copytree(os.path.join(self.origin, 'train', 'images'), "images_tmp_delete")
        for key, value in self.labels.items():
            image_file = os.path.join("images_tmp_delete", key + ".jpg")
            image = cv.imread(image_file)
            height, width = image.shape[:2]
            for rectangle in value:
                _, x, y, w, h = rectangle
                x = int(float(x)*width)
                y = int(float(y)*height)
                w = int(float(w)*width)
                h = int(float(h)*height)
                x = x - w/2
                y = y - h/2
                deleteRectangle(image, x, y, w, h)
            cv.imwrite(image_file, image)

    # run detect on the new images
    def detect(self, objects_weights_path):
        eyes = TerrarianEyes(None, objects_weights_path)
        if not os.path.exists("images_tmp_delete"):
            print("ERROR: images_tmp_delete don't exist")
            return None
        for key, value in self.labels.items():
            image_file = os.path.join("images_tmp_delete", key + ".jpg")
            image = cv.imread(image_file)
            height, width = image.shape[:2]
            objects = eyes.findObjects(image)
            for clss, rows in objects.items():
                for row in rows:
                    # centerx, centery, w, h
                    centerx = (row[0] + row[2]/2)/width
                    centery = (row[1] + row[3]/2)/height
                    w = (row[2])/width
                    h = (row[3])/height
                    value.append([str(self.classes.index(clss)), str(centerx), str(centery), str(w), str(h)])
            # save the new matches as dataset in the original labels
            label_file = os.path.join(self.origin, 'train', 'labels', key + ".txt")
            # to Test use label_file = os.path.join("images_tmp_delete",'labels', key + ".txt")
            lines = []
            for row in value:
                lines.append(f'{row[0]} {row[1]} {row[2]} {row[3]} {row[4]}\n'.replace("\n\n","\n"))
            with open(label_file, "w") as f:
                f.writelines(lines)
    
    # get a balance between classes, i.e get tghe class with less matches and 
    # deletes ramdonly the rest of matches that are above that number
    def balance(self):
        # Group classes
        classes = {}
        for key, value in self.labels.items():
            for row in value:
                if row[0] not in classes:
                    classes[row[0]] = []
                classes[row[0]].append((row, key))
        # Get class with lowest annotations
        lowest = ("", 0)
        min_length = min([len(value) for _, value in classes.items()])

        # delete ramdom boses that exceed the minimun class
        deletes = []
        for key, value in classes.items():
            while len(value) > min_length:
                deletes.append(value.pop(random.randrange(len(value))))
        
        # delete extra annotation
        if os.path.exists("new_dataset"):
            shutil.rmtree("new_dataset")
        shutil.copytree(os.path.join(self.origin), "new_dataset")

        for key, value in self.labels.items():
            i = len(deletes) - 1
            image_file = os.path.join("new_dataset", 'train', 'images' , key + ".jpg")
            label_file = os.path.join("new_dataset", 'train', 'labels' , key + ".txt")
            image = cv.imread(image_file)
            height, width = image.shape[:2]
            for rectangle,file in reversed(deletes):
                if file != key:
                    i -= 1
                    continue
                j = len(self.labels[key]) - 1
                for line in reversed(self.labels[key]):
                    if line == rectangle:
                        self.labels[key].pop(j)
                        break
                    j -= 1
                _, x, y, w, h = rectangle
                x = int(float(x)*width)
                y = int(float(y)*height)
                w = int(float(w)*width)
                h = int(float(h)*height)
                x = x - w/2
                y = y - h/2
                deleteRectangle(image, x, y, w, h)
                cv.imwrite(image_file, image)
                deletes.pop(i)
                i -= 1
            lines = []
            for row in value:
                lines.append(f'{row[0]} {row[1]} {row[2]} {row[3]} {row[4]}\n'.replace("\n\n","\n"))
            with open(label_file, "w") as f:
                f.writelines(lines)
    
    def delete_class(self, clss):
        # Group classes
        classes = {}
        for key, value in self.labels.items():
            for row in value:
                if row[0] == str(self.classes.index(clss)):
                    if row[0] not in classes:
                        classes[row[0]] = []
                    classes[row[0]].append((row, key))

        # delete ramdom boses that exceed the minimun class
        deletes = classes[str(self.classes.index(clss))]
        
        # delete annotations for that class
        if os.path.exists("new_dataset"):
            shutil.rmtree("new_dataset")
        shutil.copytree(os.path.join(self.origin), "new_dataset")

        for key, value in self.labels.items():
            i = len(deletes) - 1
            image_file = os.path.join("new_dataset", 'train', 'images' , key + ".jpg")
            label_file = os.path.join("new_dataset", 'train', 'labels' , key + ".txt")
            image = cv.imread(image_file)
            height, width = image.shape[:2]
            for rectangle,file in reversed(deletes):
                if file != key:
                    i -= 1
                    continue
                j = len(self.labels[key]) - 1
                for line in reversed(self.labels[key]):
                    if line == rectangle:
                        self.labels[key].pop(j)
                        break
                    j -= 1
                _, x, y, w, h = rectangle
                x = int(float(x)*width)
                y = int(float(y)*height)
                w = int(float(w)*width)
                h = int(float(h)*height)
                x = x - w/2
                y = y - h/2
                deleteRectangle(image, x, y, w, h)
                cv.imwrite(image_file, image)
                deletes.pop(i)
                i -= 1
            lines = []
            for row in value:
                lines.append(f'{row[0]} {row[1]} {row[2]} {row[3]} {row[4]}\n'.replace("\n\n","\n"))
            with open(label_file, "w") as f:
                f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="dataset directory")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    dataset.deleteMatches()

    objects_weights_path = os.path.join('runs', 'train', 'yolov5l6-objects', 'weights', 'best.pt')
    tiles_weights_path = os.path.join('runs', 'train', 'yolov5l6-tiles', 'weights', 'best.pt')
    #dataset.detect(objects_weights_path)
    dataset.detect(tiles_weights_path)

    dataset.balance()
    #deletes = ['anvil','arrow', 'axe', 'bar', 'bow', 'chest', 'cobwebI', 'delete', 'eye', 'furnace', 'gel', 'glowstick', 'lifeC', 'ore', 'pickaxe', 'platform', 'pot', 'potion', 'rope', 'star', 'sword', 'torchT', 'zombie']
    #deletes = ['sand','liquid', 'snow']

    #for delete in deletes:
    #    dataset.delete_class(delete)



