import numpy as np
import cv2
import os

source = '/home/annus/Desktop/Folders/staples/resized images'
destination = '/home/annus/Desktop/Folders/staples/transformed images'

# define the transformation routine
def table_transform(image):
    b = cv2.distanceTransform(src=image, distanceType=cv2.DIST_L2, maskSize=3)
    g = cv2.distanceTransform(src=image, distanceType=cv2.DIST_L1, maskSize=3)
    r = cv2.distanceTransform(src=image, distanceType=cv2.DIST_C, maskSize=3)
    trans = cv2.merge((b,g,r)) 
    #trans -= np.mean(trans)
    return trans
    
i = 0
images = os.listdir(source)
total = len(images)
print('total images = ', total)

for file_name in images:
  
    i += 1
   
    # read the image, transform and write new image
    #image = cv2.imread(os.path.join(source, file_name), cv2.THRESH_BINARY)
    image = cv2.imread(os.path.join(source, file_name), 0)   # LOAD AS GRAY SCALE IMAGE
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    tr_image = table_transform(im_bw)
    cv2.imwrite(os.path.join(destination, file_name), tr_image)
  
    # verbose
    if i % 10 == 0:
        print(i ,'/', total, ' done')




