'''
usage: python raw_get_data.py folder_name
'''

import os
import sys
import xml.etree.ElementTree as et
from PIL import Image
import shutil
import random 
import time
random.seed(int(time.time()))

source_image = '/home/annus/Desktop/Folders/full data set Images'
source_xml = '/home/annus/Desktop/Folders/proper_staples_data_set/'+sys.argv[1]
destination = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/'+sys.argv[1]
try:
    os.mkdir(destination)
except FileExistsError:
    print('deleting already existing folder!')
    shutil.rmtree(destination)
    os.mkdir(destination)

def get_images():
    i = 0
    image_list = [file for file in os.listdir(source_image) if not file.endswith('.xml')]
    xml_list = [file for file in os.listdir(source_xml) if file.endswith('.xml')]
    random.shuffle(xml_list)

    total = len(xml_list)
    split = int(0.8*total)
    print('total xmls = {}'.format(total))

    for xml in xml_list:
        tree = et.parse(os.path.join(source_xml, xml))
        root = tree.getroot()    
        depth = root.find('size').find('depth').text
        if int(depth) is not 3: 
            print('depth: {}; skipping this file:'.format(depth), xml)
            continue

        i += 1
        name, _ = os.path.splitext(xml)
        this_xml = os.path.join(source_xml, xml)
        try:
            this_image = [im_ for im_ in image_list if im_.startswith(name)][0]
        except IndexError:
            print("{} is missing", this_image)
            continue
        this_image = os.path.join(source_image, this_image)
        tree = et.parse(this_xml)

        if os.path.isfile(this_image):
            PILimg = Image.open(this_image)
            new_img = os.path.join(destination, name+'.jpg')
            if os.path.exists(new_img):
                print(name+'.jpg already exists!')
                continue
            PILimg.save(new_img)
        else:
            print("{} is missing", this_image)

        verbose = '{} / {} done'.format(i, total)
        print('\b'*len(verbose), end='', flush=True)
        print(verbose, end='')

        if i == 4:
            pass
    print('')

if __name__ == '__main__':
    get_images()














