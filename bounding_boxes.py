import xml.etree.ElementTree as et   # for reading the xml files
import cv2
import os                            # for using directory paths
import sys
import random
import time                          # for seeding the random number generator
import math
import numpy as np
import math
from scipy.spatial import distance
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

# CREDITS: Taken from @cristianpb.github.io
def rotate_box(image, bb, size_original, theta):
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

def my_own_rotation_routine(original_size, point, angle):
    '''cos = math.cos(angle)
                sin = math.sin(angle)
                rot_mat = np.matrix([[cos, -sin], [sin, cos]])'''
    (h,w) = original_size
    cx, cy = w/2., h/2.
    center = (cx, cy)
    x, y = point
    radius = distance.euclidean(point, center)
    print('radius: {:.2f}'.format(radius))
    new_point = (radius*math.cos(angle), radius*(math.sin(angle)))
    '''point = np.asarray(point) 
                new_point = np.transpose(np.matmul(rot_mat, point))
                print(new_point)'''
    print (new_point)
    return new_point

def draw_box(image, xml):
    if os.path.isfile(xml):
        tree = et.parse(xml)
        root = tree.getroot()
        objects = root.findall('.//object')
        for object_ in objects:
            x_min = int(object_.find('bndbox').find('xmin').text)
            y_min = int(object_.find('bndbox').find('ymin').text)
            x_max = int(object_.find('bndbox').find('xmax').text)
            y_max = int(object_.find('bndbox').find('ymax').text)
            
            # draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), blue, 4)
            return image

def main():
    # collect directory paths
    img_dir = sys.argv[1]
    xml_dir = sys.argv[2]
    dest_dir = sys.argv[3]

    total = len(os.listdir(img_dir))
    print('total images= ', total)
    i = 0

    for file_name in os.listdir(img_dir):
        i += 1
        # remove the filename extenstion of '.png'
        name_without_extention, _ = os.path.splitext(file_name)
        verbose = '{} / {} done'.format(i, total)
        print('\b'*len(verbose), end='', flush=True)
        print(verbose, end='')
        # read the image, without any change
        im = cv2.imread(img_dir + '/' + file_name, -1)
        # find the coordinates of b_box from the corresponding xml file
        this_xml = xml_dir + '/' + name_without_extention + '.xml'
        if os.path.isfile(this_xml):
            tree = et.parse(this_xml)
            root = tree.getroot()
            objects = root.findall('.//object')
            for object_ in objects:
                x_min = int(object_.find('bndbox').find('xmin').text)
                y_min = int(object_.find('bndbox').find('ymin').text)
                x_max = int(object_.find('bndbox').find('xmax').text)
                y_max = int(object_.find('bndbox').find('ymax').text)
                # draw the bounding box and put in the folder
                color = random.choice(colors)
                cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.imwrite(dest_dir + '/' + 'bounded_' + file_name, im)
        # verbose
    print('')

if __name__ == '__main__':
    main()

