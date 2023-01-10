from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to directory containing images to resize")
parser.add_argument("width", type=int, help="width of resized image")
parser.add_argument("height", type=int, help="height of resized image")

args = parser.parse_args()
path = args.path
width = args.width
height = args.height

if path == None or width == None or height == None:
    exit(1)

dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            print(im)
            imResize = im.resize((width,height), Image.ANTIALIAS)
            imResize.save(path + item)

resize()
