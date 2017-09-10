import xml.etree.ElementTree as et   # for reading the xml files
import os                            # for using directory paths
import sys
from PIL import Image
import numpy as np
import cv2
from transforms import table_transform as tt
import random
import time
random.seed(int(time.time()))

# collect directory paths
img_source_dir = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/'+sys.argv[1]
xml_source_dir = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/'+sys.argv[2]
train_img_dest = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/transformed/train_images'
eval_img_dest = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/transformed/eval_images'
train_transformed_dir = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/transformed/transformed_images_train'
eval_transformed_dir = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/transformed/transformed_images_eval'
train_xml_dest = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/transformed/train_xmls'
eval_xml_dest = '/home/annus/Desktop/Folders/proper_staples_data_set/outputs/fixed_imags/transformed/eval_xmls'

Size = 400, 400
xml_list = os.listdir(xml_source_dir)
image_list = [file for file in os.listdir(img_source_dir) if not file.endswith('.xml')]
xml_list = [file for file in os.listdir(xml_source_dir) if file.endswith('.xml')]
random.shuffle(xml_list) # very important part
total = len(xml_list)
split = int(0.8*total)
print('total xmls = {}'.format(total))
i = 0

for file_name in xml_list:    
    # remove the filename extenstion
    this_xml = os.path.join(xml_source_dir, file_name)
    name_without_extention, _ = os.path.splitext(file_name)

    this_img = os.path.join(img_source_dir, name_without_extention + '.png')
    if not os.path.isfile(this_img):
        this_img = os.path.join(img_source_dir, name_without_extention + '.jpg')
    if not os.path.isfile(this_img):
        this_img = os.path.join(img_source_dir, name_without_extention + '.jpeg')
    if not os.path.isfile(this_img):
        this_img = os.path.join(img_source_dir, name_without_extention + '.JPG')
    if not os.path.isfile(this_img):
        this_img = os.path.join(img_source_dir, name_without_extention + '.JPEG')
    
    if os.path.isfile(this_img) and os.path.isfile(this_xml):
        try:
            prev_tree = et.parse(this_xml)
            prev_root = prev_tree.getroot()
        except et.ParseError:
            print('error in  ', this_xml, ' file')
            continue

        depth = prev_root.find('size').find('depth').text
        if depth == '3': 
            i += 1  
        else: 
            print(file_name, "skipped")
            continue

        # remove spaces from names
        new_name = 'image_'+sys.argv[1]+'_{}'.format(i)

        # resize the image
        PILimg = Image.open(this_img)
        Pwidth, Pheight = PILimg.size
        PILimg = PILimg.resize(Size, Image.ANTIALIAS) 
        Nwidth, Nheight = PILimg.size
        if i <= split:
            new_img = os.path.join(train_img_dest, new_name) + '.jpg'
        else:
            new_img = os.path.join(eval_img_dest, new_name) + '.jpg'
        PILimg.save(new_img)    

        # create a new xml file and just plug in the existing coordinate values into it                                                         
        root = et.Element('annotation')
        et.SubElement(root, 'folder').text = "Staples"
        et.SubElement(root, 'filename').text = new_name + '.jpg'
 
        source = et.SubElement(root, 'source')
        et.SubElement(source, 'database').text = "Staples data set"
        et.SubElement(source, 'annotation').text = "Staples"
        et.SubElement(source, 'image').text = "flickr"

        size = et.SubElement(root, 'size')
        imag = cv2.imread(new_img, 0)   # LOAD AS GRAY SCALE IMAGE
        (thresh, im_bw) = cv2.threshold(imag, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        tr_image = tt(im_bw, 0.3)
        if i <= split:
            cv2.imwrite(os.path.join(train_transformed_dir, new_name) + '.jpg', tr_image)
        else:
            cv2.imwrite(os.path.join(eval_transformed_dir, new_name) + '.jpg', tr_image)
        height, width, channels = (400, 400, 3)
        et.SubElement(size, 'width').text = str(width)
        et.SubElement(size, 'height').text = str(height)
        et.SubElement(size, 'depth').text = str(channels)

        et.SubElement(root, 'segmented').text = str(0)

        # get all tables
        objects = prev_root.findall('.//object')
        for child in objects:
            _object = et.SubElement(root, 'object')
            et.SubElement(_object, 'name').text = 'table'
            et.SubElement(_object, 'pose').text = 'Frontal'
            et.SubElement(_object, 'truncated').text = str(0)
            et.SubElement(_object, 'occluded').text = str(0)

            bnd_box = et.SubElement(_object, 'bndbox')
            x_min = int(child.find('bndbox').find('xmin').text)
            y_min = int(child.find('bndbox').find('ymin').text)
            x_max = int(child.find('bndbox').find('xmax').text)
            y_max = int(child.find('bndbox').find('ymax').text)

            x_min = int(x_min*Nwidth/Pwidth) 
            y_min = int(y_min*Nheight/Pheight)
            x_max = int(x_max*Nwidth/Pwidth) 
            y_max = int(y_max*Nheight/Pheight)
            et.SubElement(bnd_box, 'xmin').text = str(x_min)
            et.SubElement(bnd_box, 'ymin').text = str(y_min)
            et.SubElement(bnd_box, 'xmax').text = str(x_max)
            et.SubElement(bnd_box, 'ymax').text = str(y_max)

            et.SubElement(_object, 'difficult').text = str(0)

        tree = et.ElementTree(root)
        if i <= split:
            tree.write(os.path.join(train_xml_dest, new_name) + '.xml')
        else:
            tree.write(os.path.join(eval_xml_dest, new_name) + '.xml')
    
    else:
        print(this_img, ' not found')    

    # verbose
    verbose = '{} in training data'.format(i) 
    if i > split: verbose = '{} in eval data'.format(i-split) 
    print('\b'*len(verbose), end='', flush=True)
    print(verbose, end='')
    if i is split: 
        print('\b'*len(verbose), end='', flush=True)
        print('')

print('')


        