import xml.etree.ElementTree as et   # for reading the xml files
import os                            # for using directory paths
import cv2
import numpy as np

# collect directory paths
xml_dir = 'UNLV/Annotations-xml'
img_dir = 'UNLV/unlv-table-png'
dest_dir = 'editted xmls'
xmls = os.listdir(xml_dir)
total = len(xmls)
print('total xmls= ', total)            
i = 0

for file_name in xmls:

    i += 1
    name, _ = os.path.splitext(file_name)
    img_name = os.path.join(img_dir, name + '.png') 

    if os.path.isfile(img_name):
        # start with a new xml file and just plug in the existing coordinate values into it                                                         
        root = et.Element('annotation')
        et.SubElement(root, 'folder').text = "UNLV"
        et.SubElement(root, 'filename').text = name + '.jpg'

        source = et.SubElement(root, 'source')
        et.SubElement(source, 'database').text = "UNLV data set"
        et.SubElement(source, 'annotation').text = "UNLV"
        et.SubElement(source, 'image').text = "flickr"

        size = et.SubElement(root, 'size')
        img = cv2.imread(img_name, -1)
        height, width, channels = img.shape
        et.SubElement(size, 'width').text = str(width)
        et.SubElement(size, 'height').text = str(height)
        et.SubElement(size, 'depth').text = str(channels)

        et.SubElement(root, 'segmented').text = str(0)

        # run through all of the tables
        prev_tree = et.parse(os.path.join(xml_dir, file_name))
        prev_root = prev_tree.getroot()
        
        _object = et.SubElement(root, 'object')
        et.SubElement(_object, 'name').text = 'table'
        et.SubElement(_object, 'pose').text = 'Frontal'
        et.SubElement(_object, 'truncated').text = str(0)
        et.SubElement(_object, 'occluded').text = str(0)

        bnd_box = et.SubElement(_object, 'bndbox')
        
        x_min = int(prev_root.find('object').find('bndbox').find('xmin').text)
        y_min = int(prev_root.find('object').find('bndbox').find('ymin').text)
        x_max = int(prev_root.find('object').find('bndbox').find('xmax').text)
        y_max = int(prev_root.find('object').find('bndbox').find('ymax').text)
        et.SubElement(bnd_box, 'xmin').text = str(x_min)
        et.SubElement(bnd_box, 'ymin').text = str(y_min)
        et.SubElement(bnd_box, 'xmax').text = str(x_max)
        et.SubElement(bnd_box, 'ymax').text = str(y_max)

        et.SubElement(_object, 'difficult').text = str(0)

        tree = et.ElementTree(root)
        tree.write(os.path.join(dest_dir, name) + '.xml')

    else:
        print(name + '.png', ' not found')

    # verbose
    if i % 10 == 0:
        print(i ,'/', total, ' converted')

    #if i == 1:
    #    break












