
"""
Author: Annus Zulfiqar
Program: data_augmentation.py 
Date: 07-09-17
Usage: Meant for doing some handy data augmentation on small object detection datasets 
Calling: Pass it the images, annotations, their destination folders and one more folder
         for seeing the results
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

def set_size(root, new_size):
    height, width, _ = new_size
    root.find('size').find('width').text = str(width)
    root.find('size').find('height').text = str(height)
    return root

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
    for object_, coord in zip(objects, coords):
        object_.find('bndbox').find('xmin').text = str(coord[0])
        object_.find('bndbox').find('ymin').text = str(coord[1])
        object_.find('bndbox').find('xmax').text = str(coord[2])
        object_.find('bndbox').find('ymax').text = str(coord[3])
    return root


def set_file_name(root, file_name):
    root.find('filename').text = file_name
    return root


# 0:x_min, 1:y_min, 2:x_max, 3:y_max
def get_4_corners(coords):
    return [tuple((coord[0], coord[1], coord[2], coord[1],
        coord[2], coord[3], coord[0], coord[3])) for coord in coords]


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


# CREDITS: taken from @cristianpb.github.io, editted by me
def rotate_box(coords, size_original, Matrix=None, theta=None):   
    all_corners = get_4_corners(coords)
    new_coords = []
    for corners in all_corners:
        (w, h) = size_original
        (cx, cy) = ((w-1) // 2, (h-1) // 2)
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0) if Matrix is None else Matrix
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += ((nW-1) / 2) - cx
        M[1, 2] += ((nH-1) / 2) - cy
        #1 Prepare the vector to be transformed
        v = [corners[0],corners[1],1]
        calculated_1 = np.dot(M,v)
        v = [corners[2],corners[3],1]
        calculated_2 = np.dot(M,v)
        v = [corners[4],corners[5],1]
        calculated_3 = np.dot(M,v)
        v = [corners[6],corners[7],1]
        calculated_4 = np.dot(M,v)
        x_min = int(min(calculated_1[0], calculated_2[0], calculated_3[0], calculated_4[0]))
        y_min = int(min(calculated_1[1], calculated_2[1], calculated_3[1], calculated_4[1]))
        x_max = int(max(calculated_1[0], calculated_2[0], calculated_3[0], calculated_4[0]))
        y_max = int(max(calculated_1[1], calculated_2[1], calculated_3[1], calculated_4[1]))
        x_min = max(0, x_min)
        x_max = min(x_max, nW-1)
        y_min = max(0, y_min)
        y_max = min(y_max, nH-1)
        new_coords.append(tuple((x_min, y_min, x_max, y_max)))
    return new_coords


def draw_box(image, coords, pen_width, color=red):
    for coord in coords:
        x_min, y_min, x_max, y_max = coord
        # draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
            color, pen_width)
    return image


def check_rotation(image, xml_root, angle=0.0):
    coords = get_coords(xml_root)
    rows, cols, depth = get_shape(xml_root)
    new_image = image #draw_box(image=image, coords=coords, pen_width=2, color=blue)
    M = cv2.getRotationMatrix2D(((rows-1)/2, (cols-1)/2), angle, 1)
    dst = imutils.rotate_bound(new_image, angle)    
    new_coords = rotate_box(coords=coords, size_original=(rows, cols), theta=-angle)
    new_image = draw_box(image=dst, coords=new_coords, pen_width=2, color=red)
    cv2.imshow('new image with bounding box', dst)
    cv2.waitKey(0)
    return dst, new_coords


def check_translation(x=0.0, y=0.0):
    image = cv2.imread('00P3800000d2wgPEAQ#Mason List 11-1.jpg', 1)
    xml_root = et.parse('00P3800000d2wgPEAQ#Mason List 11-1.xml').getroot()
    coords = get_coords(xml_root)
    rows, cols, depth = get_shape(xml_root)
    new_image = draw_box(image=image, coords=coords, pen_width=2, color=blue)
    M = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(image, M, (rows, cols))
    new_coords = rotate_box(coords=coords, size_original=(rows, cols), Matrix=M)
    new_image = draw_box(image=dst, coords=new_coords, pen_width=2, color=red)
    cv2.imshow('new image with bounding box', dst)


def rotate(image, xml_root, theta=0.0):
    coords = get_coords(xml_root)
    rows, cols, depth = get_shape(xml_root)
    M = cv2.getRotationMatrix2D(((rows-1)/2, (cols-1)/2), theta, 1)
    new_image = imutils.rotate_bound(image, theta)    
    new_coords = rotate_box(coords=coords, size_original=(rows, cols), theta=-theta)
    return new_image, new_coords


def translate(image, xml_root, displacement_tuple=(0.0, 0.0)):
    x, y = displacement_tuple
    coords = get_coords(xml_root)
    rows, cols, depth = get_shape(xml_root)
    M = np.float32([[1,0,x],[0,1,y]])
    new_image = cv2.warpAffine(image, M, (rows, cols))
    new_coords = rotate_box(coords=coords, size_original=(rows, cols), Matrix=M)
    return new_image, new_coords


def perform_augmentation(images_folder, xmls_folder, im_dest_folder,
    xmls_dest_folder, rot=True, trans=True, bounded_folder=None):
    xmls_list = os.listdir(xmls_folder)
    total = len(xmls_list)
    log('xmls list acquired')
    stroke_width = 2

    for idx, xml in enumerate(xmls_list, 1):
        name, _ = os.path.splitext(xml)
        xml_path = os.path.join(xmls_folder, xml)
        image_path = os.path.join(images_folder, name+'.jpg')
        if not os.path.exists(image_path): 
            log('image: {} not found'.format('{}.jpg'.format(name)))
            continue
        this_tree = et.parse(xml_path)
        this_xml = this_tree.getroot()
        root = this_tree.getroot()
        coords = get_coords(root)

        this_image = cv2.imread(image_path, 1)
        # check the shape of the image using its xml
        _, _, depth_check = this_image.shape
        if depth_check is not 3:
            log('incorrect image: {}'.format(name))
            continue
        this_xml = et.parse(xml_path).getroot()
        log('on image ({} of {})'.format(idx, total), cute=True)

        # rotate the image in 2 directions
        if rotate:
            for count in range(5):
                i = random.randint(1, 7)
                j = random.randint(-7, -1)
                # save the original image once
                theta = i
                theta = 0 if count is 0 else i
             
                new_image, new_coords = rotate(image=this_image, xml_root=this_xml, theta=float(theta))
                new_name = 'rot_{}_{}'.format(name, str(theta))
                cv2.imwrite(os.path.join(im_dest_folder, new_name+'.jpg'), new_image)
                new_root = set_coords(root, new_coords)
                new_root = set_file_name(new_root, new_name+'.jpg')
                new_shape = new_image.shape
                new_root = set_size(new_root, new_shape)
                new_tree = et.ElementTree(new_root)
                new_tree.write(os.path.join(xmls_dest_folder, new_name+'.xml'))
                if bounded_folder:
                    bounded_image = draw_box(image=new_image, coords=new_coords, 
                        pen_width=stroke_width, color=random.choice(colors))
                    cv2.imwrite(os.path.join(bounded_folder, new_name+'.jpg'), bounded_image)
                if count is 0: continue

                new_image, new_coords = rotate(image=this_image, xml_root=this_xml, theta=float(j))
                new_name = 'rot_{}_{}'.format(name, str(j))
                cv2.imwrite(os.path.join(im_dest_folder, new_name+'.jpg'), new_image)
                new_root = set_coords(root, new_coords)
                new_root = set_file_name(new_root, new_name+'.jpg')
                new_shape = new_image.shape
                new_root = set_size(new_root, new_shape)                
                new_tree = et.ElementTree(new_root)
                new_tree.write(os.path.join(xmls_dest_folder, new_name+'.xml'))
                if bounded_folder:
                    bounded_image = draw_box(image=new_image, coords=new_coords, 
                        pen_width=4, color=random.choice(colors))
                    cv2.imwrite(os.path.join(bounded_folder, new_name+'.jpg'), bounded_image)                    

        # translate the image in all 4 directions
        if translate:
            for count in range(3): 
                x = random.randint(20, 50) + random.randint(1, 10)
                y = random.randint(20, 50) + 2*random.randint(1, 10)
                new_image, new_coords = translate(image=this_image, xml_root=this_xml, 
                    displacement_tuple=(float(x), float(y)))
                new_name = 'trans_1_{}_{}_{}'.format(name, str(x), str(y))
                cv2.imwrite(os.path.join(im_dest_folder, new_name+'.jpg'), new_image)
                new_root = set_coords(root, new_coords)
                new_root = set_file_name(new_root, new_name+'.jpg')
                new_shape = new_image.shape
                new_root = set_size(new_root, new_shape)                
                new_tree = et.ElementTree(new_root)
                new_tree.write(os.path.join(xmls_dest_folder, new_name+'.xml'))
                if bounded_folder:
                    bounded_image = draw_box(image=new_image, coords=new_coords, 
                        pen_width=stroke_width, color=random.choice(colors))
                    cv2.imwrite(os.path.join(bounded_folder, new_name+'.jpg'), bounded_image)

                x = random.randint(20, 50) + 2*random.randint(-10, 1)
                y = random.randint(-50, -20) + 2*random.randint(-10, 1)
                new_image, new_coords = translate(image=this_image, xml_root=this_xml, 
                    displacement_tuple=(float(x), float(y)))              
                new_name = 'trans_2_{}_{}_{}'.format(name, str(x), str(y))                          
                cv2.imwrite(os.path.join(im_dest_folder, new_name+'.jpg'), new_image)
                new_root = set_coords(root, new_coords)
                new_root = set_file_name(new_root, new_name+'.jpg')
                new_shape = new_image.shape
                new_root = set_size(new_root, new_shape)                
                new_tree = et.ElementTree(new_root)
                new_tree.write(os.path.join(xmls_dest_folder, new_name+'.xml'))
                if bounded_folder:
                    bounded_image = draw_box(image=new_image, coords=new_coords, 
                        pen_width=stroke_width, color=random.choice(colors))
                    cv2.imwrite(os.path.join(bounded_folder, new_name+'.jpg'), bounded_image)

                x = random.randint(-50, 20) + 2*random.randint(1, 10)
                y = random.randint(20, 50) + random.randint(-10, -1)
                new_image, new_coords = translate(image=this_image, xml_root=this_xml, 
                    displacement_tuple=(float(x), float(y)))                    
                new_name = 'trans_3_{}_{}_{}'.format(name, str(x), str(y))                
                cv2.imwrite(os.path.join(im_dest_folder, new_name+'.jpg'), new_image)
                new_root = set_coords(root, new_coords)
                new_root = set_file_name(new_root, new_name+'.jpg')
                new_shape = new_image.shape
                new_root = set_size(new_root, new_shape)                
                new_tree = et.ElementTree(new_root)
                new_tree.write(os.path.join(xmls_dest_folder, new_name+'.xml'))
                if bounded_folder:
                    bounded_image = draw_box(image=new_image, coords=new_coords, 
                        pen_width=stroke_width, color=random.choice(colors))
                    cv2.imwrite(os.path.join(bounded_folder, new_name+'.jpg'), bounded_image)

                x = random.randint(-50, -20) + 2*random.randint(-10, 1)
                y = random.randint(-50, -20) + random.randint(1, 10)
                new_image, new_coords = translate(image=this_image, xml_root=this_xml, 
                    displacement_tuple=(float(x), float(y)))                    
                new_name = 'trans_4_{}_{}_{}'.format(name, str(x), str(y))                
                cv2.imwrite(os.path.join(im_dest_folder, new_name+'.jpg'), new_image)
                new_root = set_coords(root, new_coords)
                new_root = set_file_name(new_root, new_name+'.jpg')
                new_shape = new_image.shape
                new_root = set_size(new_root, new_shape)                
                new_tree = et.ElementTree(new_root)
                new_tree.write(os.path.join(xmls_dest_folder, new_name+'.xml'))
                if bounded_folder:
                    bounded_image = draw_box(image=new_image, coords=new_coords, 
                        pen_width=stroke_width, color=random.choice(colors))
                    cv2.imwrite(os.path.join(bounded_folder, new_name+'.jpg'), bounded_image)
        time.sleep(0.01)

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
    parser.add_argument('--bf', '--bounded_dest_folder', type=str, dest='bounded_dest_folder',
        help='bounded destination folder')
    args = parser.parse_args()

    images_folder = args.image_folder 
    xmls_folder = args.xml_folder
    images_dest = args.image_dest_folder
    xmls_dest = args.xml_dest_folder
    bounded_folder = args.bounded_dest_folder

    perform_augmentation(images_folder=images_folder, xmls_folder=xmls_folder, 
        im_dest_folder=images_dest, xmls_dest_folder=xmls_dest, bounded_folder=bounded_folder)


'''for i in range(4000):
                check_rotation(float(i))
                time.sleep(0.001)'''
if __name__ == '__main__':
    log('Entering main routine') 
    '''image = cv2.imread('image_layout_1_39.jpg', 1)
                xml_root = et.parse('image_layout_1_39.xml').getroot()
                for i in range(4000):
                    check_rotation(image, xml_root, float(i))
                    if cv2.waitKey(0) == 0xFF & ord('q'): break
                    time.sleep(0.001)'''
    '''image = cv2.imread("/home/annus/Desktop/Folders/project/00P3800000d2wj8EAA#Mason List 7-1.jpg", 1)
                print(image.shape)'''
    main()







