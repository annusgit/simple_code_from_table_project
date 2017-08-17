import xml.etree.ElementTree as et   # for reading the xml files
import cv2
import os                            # for using directory paths
import random
import time                          # for seeding the random number generator

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

# collect directory paths
img_dir = '/home/annus/Desktop/Folders/staples/resized images'
xml_dir = '/home/annus/Desktop/Folders/staples/xmls'
dest_dir = '/home/annus/Desktop/Folders/staples/bounded images'

total = len(os.listdir(img_dir))
print('total images= ', total)
i = 0

for file_name in os.listdir(img_dir):
  
    i += 1
    
    # remove the filename extenstion of '.png'
    name_without_extention, _ = os.path.splitext(file_name)

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
            cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, 1)
        cv2.imwrite(dest_dir + '/' + 'bounded_' + file_name, im)
        
    # verbose
    if i % 10 == 0:
        print(i ,'/', total, ' done')
        



