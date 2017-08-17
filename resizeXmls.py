import xml.etree.ElementTree as et   # for reading the xml files
import os                            # for using directory paths
from PIL import Image
import numpy as np
import cv2

# collect directory paths
#img_dir = 'UNLV/jpeg transformed images'
xml_source_dir = '/home/annus/Desktop/Folders/labeled data'
img_source_dir = '/home/annus/Desktop/Folders/full data set Images'
#checksource_dir = 'UNLV/unlv-table-png'
#checkdest_dir = 'UNLV/checkimg'
imgdest_dir = '/home/annus/Desktop/Folders/staples/resized images'
xmldest_dir = '/home/annus/Desktop/Folders/staples/xmls'
xml_list = os.listdir(xml_source_dir)

xmls = [xml for xml in xml_list if xml.endswith('.xml')]

#imgs = os.listdir(img_dir)
total = len(xmls)
print('total xmls = ', total)
scale = 400
Size = scale, scale
i = 0

for file_name in xml_list:
  
    i += 1
    
    # remove the filename extenstion of '.jpg'
    this_xml = os.path.join(xml_source_dir, file_name)
    name_without_extention, _ = os.path.splitext(file_name)

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
            print('missing ', this_xml, ' file')
            continue

        # resize the image
        PILimg = Image.open(this_img)
        Pwidth, Pheight = PILimg.size
        PILimg.thumbnail(Size, Image.ANTIALIAS)
        Nwidth, Nheight = PILimg.size
        new_img = os.path.join(imgdest_dir, name_without_extention) + '.jpg'
        PILimg.save(new_img)

        #PILimg = Image.open(os.path.join(checksource_dir, name_without_extention) + '.png')
        #PILimg.thumbnail(Size, Image.ANTIALIAS)
        #PILimg.save(os.path.join(checkdest_dir, name_without_extention) + '.jpg')        

        # start with a new xml file and just plug in the existing coordinate values into it                                                         
        root = et.Element('annotation')
        et.SubElement(root, 'folder').text = "Staples"
        et.SubElement(root, 'filename').text = name_without_extention + '.jpg'

        source = et.SubElement(root, 'source')
        et.SubElement(source, 'database').text = "Staples data set"
        et.SubElement(source, 'annotation').text = "Staples"
        et.SubElement(source, 'image').text = "flickr"

        size = et.SubElement(root, 'size')
        img = cv2.imread(new_img, -1)
        try:
            height, width, channels = img.shape
        except ValueError:
            print(this_img, ' has a problem')
        et.SubElement(size, 'width').text = str(width)
        et.SubElement(size, 'height').text = str(height)
        et.SubElement(size, 'depth').text = str(channels)

        et.SubElement(root, 'segmented').text = str(0)

        # get all tables
        objects = prev_root.findall('.//object')
        for child in objects:
            # child = prev_root[k]
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
        tree.write(os.path.join(xmldest_dir, name_without_extention) + '.xml')
    
    else:
        print(this_img, ' not found')    

    # verbose
    if i % 10 == 0:
        print(i ,'/', total, ' done')

    #if i == 4:
    #    break



        