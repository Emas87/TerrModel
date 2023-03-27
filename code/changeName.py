# python D:\Desktop\changeName.py -regexp "-" -sub "_" -path .

import argparse
import os
import re
parser = argparse.ArgumentParser()
parser.add_argument("-path", help="new value")
args = parser.parse_args()

files = os.listdir(args.path)
for i in range(0,len(files)):
    NewFile = "grass" + str(i)
    file = os.path.join(args.path, files[i])
    NewFile = os.path.join(args.path, NewFile + ".png")
    if file != NewFile:
        os.rename(file, NewFile)
        print("os.rename(" + file + ", " + NewFile + ")")

