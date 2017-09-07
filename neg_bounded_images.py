import cv2
import os                            # for using directory paths
import random
import numpy as np                   # for randint 
import time                          # for seeding the random number generator

random.seed(int(time.time()))
np.random.seed(int(time.time()))
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
img_dir = 'UNLV/unlv-table-png'
dest_dir = 'negative bounded images'

total = len(os.listdir(img_dir))
print('total images= ', total)
i = 0

for file_name in os.listdir(img_dir):
  
    i += 1
    
    # remove the filename extenstion of '.png'
    name_without_extention, _ = os.path.splitext(file_name)

    # read the image, without any change
    im = cv2.imread(img_dir + '/' + file_name, -1)

    # generate random(probably wrong) coordinates
    x_min = np.random.randint(0, 2000)
    y_min = np.random.randint(0, 2000)
    x_max = np.random.randint(0, 2000) + 1000
    y_max = np.random.randint(0, 2000) + 1000

    # draw the bounding box and put in the folder
    color = random.choice(colors)
    cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, 10)
    cv2.imwrite(dest_dir + '/' + 'bounded_' + file_name, im)
    
    # verbose
    if i % 10 == 0:
        print(i ,'/', total, ' done')


