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

def get_angle(A, B, center=(0, 0)):
    '''x1, x2 = A[0]-center[0], B[0]-center[0]  
                y1, y2 = A[1]-center[1], B[1]-center[1]  
                a = math.sqrt(x1**2+y1**2)
                b = math.sqrt(x2**2+y2**2)
                dot_product = x1*x2+y1*y2
                return math.acos(dot_product/(a*b))
            '''
    return math.atan2(A[1]-B[1], A[0]-B[0])

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


def my_rotate_box(coords, center, angle, M, size):
    new_coords = []
    for coord in coords:
        print(coord)
        x_min, y_min, x_max, y_max = coord
        all_4_coords = np.asarray([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                                    dtype=np.float32)
        n_coords = cv2.warpAffine(all_4_coords, M, size)
        new_coords.append(n_coords)

        # log('{} {} {} {}'.format(x_min, y_min, x_max, y_max), clause='coordinates')
        new_coords.append(tuple((min(x_min, x_max), min(y_min, y_max),
            max(x_min, x_max), max(y_min, y_max))))

        '''cx = int(center[0])
        cy = int(center[1])
        horizontal = (center[0]+10, center[1])
        radius = distance.euclidean(tuple(center), tuple((x_min, y_min)))
        theta = get_angle(horizontal, (x_min, y_min))
        theta = theta + math.radians(angle)
        x_min = int(x_min*math.cos(theta))-int(y_min*math.sin(theta))+cx
        y_min = int(y_min*math.cos(theta))+int(x_min*math.sin(theta))+cy
        radius = distance.euclidean(tuple(center), tuple((x_max, y_max)))
        theta = get_angle(horizontal, (x_max, y_max))
        theta = theta + math.radians(angle)
        x_max = int(x_max*math.cos(theta))-int(y_max*math.sin(theta))+cx
        y_max = int(y_max*math.cos(theta))+int(x_max*math.sin(theta))+cy
        print(x_min, y_min, x_max, y_max)'''
        '''new_coords.append(tuple((min(x_min, x_max), min(y_min, y_max),
            max(x_min, x_max), max(y_min, y_max))))'''
    return new_coords


# CREDITS: Taken from @cristianpb.github.io
def rotate_box(bb, size_original, theta):
    (h, w) = size_original
    (cx, cy) = (w // 2, h // 2)
    new_bb = list(bb)
    for i, coord in enumerate(bb):
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
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb


def draw_box(image, coords, pen_width):
    for coord in coords:
        x_min, y_min, x_max, y_max = coords[0]
        # draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), random.choice(colors), pen_width)
    return image


def check(angle=0):
    image = cv2.imread('00P3800000cqvteEAA#Amazon 5.jpg', 1)
    xml_root = et.parse('00P3800000cqvteEAA#Amazon 5.xml').getroot()
    coords = get_coords(xml_root)
    rows, cols, depth = get_shape(xml_root)
    center = ((rows-1)/2, (cols-1)/2)

    image = draw_box(image=image, coords=coords, pen_width=5)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols,rows))
    
    # image = imutils.rotate_bound(image, angle)
    new_coords = []
    for coord in coords:
        first = rotate_bbox(coords=(coords), center=center,
            angle=angle, M=M, size=(cols, rows))
        '''second = rotate_box(coords=(coords[2], coords[3]), center=center,
                                    angle=angle, M=M, size=(cols, rows))'''
        new_coords.append(first)                                    
    new_image = draw_box(image=image, coords=new_coords, pen_width=5)
    cv2.imshow('new image with bounding box', new_image)
    cv2.waitKey()


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
    check(21)










