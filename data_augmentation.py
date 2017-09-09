"""
Author: Annus Zulfiqar
Program: data_augmentation.py 
Date: 07-09-17
Usage: Meant for doing some handy data augmentation on small object detection datasets 
Calling: Pass it the images, annotations and the destination folders  
"""

import os 
import cv2
import math
import time
import random
import imutils
import argparse
import numpy as np
from log import log
from scipy.spatial import distance
import xml.etree.ElementTree as et
log('Imports successful!')
random.seed(int(time.time()))

# for colorful boxes!
red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
orange = (255, 69, 0)
gold = (255, 215, 0)
aqua = (0, 255, 255)
pink = (255, 105, 180)
colors = [blue, green, red, orange, gold, aqua, pink]
random.shuffle(colors)


# some handy xml functions
def get_shape(root):
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    depth = int(root.find('size').find('depth').text)
    return (width, height, depth)


def get_coords(root):
    coords = []
    objects = root.findall('.//object')
    for object_ in objects:
        x_min = int(object_.find('bndbox').find('xmin').text)
        y_min = int(object_.find('bndbox').find('ymin').text)
        x_max = int(object_.find('bndbox').find('xmax').text)
        y_max = int(object_.find('bndbox').find('ymax').text)
        coords.append(tuple((x_min, y_min, x_max, y_max)))
    return coords

    
def set_coords(root, coords):
    objects = root.findall('.//object')
    for object_ in objects:
        x_min = int(object_.find('bndbox').find('xmin').text)
        y_min = int(object_.find('bndbox').find('ymin').text)
        x_max = int(object_.find('bndbox').find('xmax').text)
        y_max = int(object_.find('bndbox').find('ymax').text)
        coords.append(tuple((x_min, y_min, x_max, y_max)))
    return root


# 0:x_min, 1:y_min, 2:x_max, 3:y_max
def get_4_corners(coords):
    return [tuple((coord[0], coord[1], coord[2], coord[1],
        coord[2], coord[3], coord[0], coord[3])) for coord in coords]


# translates the image by x, as well as the bounding_box in the direction passed as
# a keyword arguement
def translate(**kwargs):
    if len(kwargs) is not 4:
        raise ValueError('{} args passed. 4 needed'.format(len(locals())))
    directions = ['up', 'down', 'right', 'left']
    for key in kwargs:
        if key is 'image':
            image_array = kwargs[key]
        elif key is 'coords':
            coords = kwargs[key]
            x_min, y_min, x_max, y_max = map(int, coords)
        elif key is 'displacement':
            x = kwargs[key]
        elif key is 'direction':
            direction = kwargs[key]
            if direction not in directions:
                raise ValueError('Invalid value \'{}\' in direction.\n\
Possible Values: {}'.format(direction, directions))
        else: 
            raise KeyError('Invalid key \'{}\' keys'.format(key))
        

def rotate_image(im):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 800)
    cv2.moveWindow('image', 0, 0)
    cv2.imshow('image', im)
    new_image = im.copy()
    saved = False
    read_key = cv2.waitKey(33)
    while not saved:
        while read_key == -1: read_key = cv2.waitKey(33)
        if read_key is 0xFF & ord('r'):
            new_image = imutils.rotate(new_image, -90)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 800, 800)
            cv2.moveWindow('image', 0, 0)
            cv2.imshow('image', new_image)
            read_key = -1
            time.sleep(0.1)
        if read_key is 0xFF & ord('s'):
            cv2.imwrite(os.path.join(dest_folder, image_name), new_image)
            verbose = 'image#'+str(i+1)+' saved'
            log(verbose, cute=True)
            saved = True
            read_key = -1
            time.sleep(0.1)
    log()


# CREDITS: Taken from @cristianpb.github.io
def rotate_box(coord, size_original, theta):
    (w, h) = size_original
    (cx, cy) = (w // 2, h // 2)
    # for i, coord in enumerate(bb):
    # opencv calculates standard transformation matrix
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    # Grab  the rotation components of the matrix)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    #1 Prepare the vector to be transformed
    v = [coord[0],coord[1],1]
    # Perform the actual rotation and return the image
    calculated_1 = np.dot(M,v)
    #2 Prepare the vector to be transformed
    v = [coord[2],coord[3],1]
    # Perform the actual rotation and return the image
    calculated_2 = np.dot(M,v)
    #1 Prepare the vector to be transformed
    v = [coord[4],coord[5],1]
    # Perform the actual rotation and return the image
    calculated_3 = np.dot(M,v)
    #2 Prepare the vector to be transformed
    v = [coord[6],coord[7],1]
    # Perform the actual rotation and return the image
    calculated_4 = np.dot(M,v)
    x_min = min(calculated_1[0], calculated_2[0], calculated_3[0], calculated_4[0])
    y_min = min(calculated_1[1], calculated_2[1], calculated_3[1], calculated_4[1])
    x_max = max(calculated_1[0], calculated_2[0], calculated_3[0], calculated_4[0])
    y_max = max(calculated_1[1], calculated_2[1], calculated_3[1], calculated_4[1])
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def draw_box(image, coords, pen_width):
    for coord in coords:
        # print(type(coord))
        x_min, y_min, x_max, y_max = coord
        # draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
            random.choice(colors), pen_width)
    return image


def check(angle=0.0):
    image = cv2.imread('00P3800000cqvteEAA#Amazon 5.jpg', 1)
    xml_root = et.parse('00P3800000cqvteEAA#Amazon 5.xml').getroot()
    coords = get_coords(xml_root)
    rows, cols, depth = get_shape(xml_root)
    center = ((rows-1)/2, (cols-1)/2)

    new_image = draw_box(image=image, coords=coords, pen_width=2)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = imutils.rotate_bound(new_image, angle)
    
    # image = imutils.rotate_bound(image, angle)
    all_corners = get_4_corners(coords)
    new_coords = [rotate_box(coord=corners, size_original=(rows, cols),
        theta=-angle) for corners in all_corners]
    new_image = draw_box(image=dst, coords=new_coords, pen_width=2)
    cv2.imshow('new image with bounding box', dst)
    # cv2.waitKey()


def main():
    parser = argparse.ArgumentParser(description='this file augments obj_det dataset')
    parser.add_argument('--if', '--image_folder', type=str, dest='image_folder',
        help='image folder')
    parser.add_argument('--xf', '--xml_folder', type=str, dest='xml_folder',
        help='xml folder')
    parser.add_argument('--idf', '--image_dest_folder', type=str, dest='image_dest_folder',
        help='image destination folder')
    parser.add_argument('--xdf', '--xml_dest_folder', type=str, dest='xml_dest_folder',
        help='xml destination folder')
    args = parser.parse_args()

    images_folder = args.image_folder 
    xmls_folder = args.xml_folder
    images_dest = args.image_dest_folder
    xmls_dest = args.xml_dest_folder

    # create list of all xmls
    roots_list = [(xml.replace('.xml', ''), et.parse(os.path.join(xmls_folder, xml)).getroot())
        for xml in os.listdir(xmls_folder)]
    log('All xml roots acquired')
    for idx, (file_name, root) in enumerate(roots_list, 1):
        log('on image {} ({} of {})'.format(file_name, idx, len(roots_list)), cute=True)
        x_min = root.find('object').find('bndbox').find('xmin').text
        y_min = root.find('object').find('bndbox').find('ymin').text
        x_max = root.find('object').find('bndbox').find('xmax').text
        y_max = root.find('object').find('bndbox').find('ymax').text

        translate(image=cv2.imread(os.path.join(images_folder, file_name+'.jpg'), 1),
            coords=(x_min, y_min, x_max, y_max), displacement=10, direction='right')
    log()

if __name__ == '__main__':
    log('Entering main routine') 
    i = 0
    while True:
        check(i)
        log('{}'.format(i), clause='rotation#', cute=True)
        time.sleep(0.001)
        i += 1
        if cv2.waitKey(33) == 0xFF & ord('q'):
            break
    log()











