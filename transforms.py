import numpy as np
import cv2
import os
from pylab import array, uint8
source = '/home/annus/Desktop/Folders/project/UNLV/staples_separated_eval_data/resized_images'
destination = '/home/annus/Desktop/Folders/project/UNLV/'

# define the transformation routine
def table_transform(original, contrast_x=None):
    (thresh, image) = cv2.threshold(original, 128, 255, cv2.THRESH_BINARY)
    b = cv2.distanceTransform(src=image, distanceType=cv2.DIST_L1, maskSize=3)
    g = cv2.distanceTransform(src=image, distanceType=cv2.DIST_L2, maskSize=3)
    r = cv2.distanceTransform(src=image, distanceType=cv2.DIST_C, maskSize=3)
    cv2.normalize(b,  b, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(g, g, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(r, r, 0, 255, cv2.NORM_MINMAX)
    trans = cv2.merge((b,g,r)) 
    if not contrast_x: return trans 
    # Parameters for manipulating image contrast
    phi = 1
    theta = 1
    maxIntensity = 255.0 # depends on dtype of image data
    # Increase intensity such that
    # dark pixels become much brighter, 
    # bright pixels become slightly bright
    newImage = (maxIntensity/phi)*(trans/(maxIntensity/theta))**contrast_x
    newImage = array(newImage, dtype=uint8)
    return newImage

def test_transform_definition():
    image = cv2.imread('UNLV/table.png', 0)   # LOAD AS GRAY SCALE IMAGE
    tr_image = table_transform(image)
    cv2.imwrite(os.path.join('UNLV/without_con_0101_003.jpg'), tr_image)
    print(np.max(tr_image), np.min(tr_image))

    tr_image = table_transform(image, 0.2)
    cv2.imwrite(os.path.join('UNLV/x_0101_003.jpg'), tr_image)
    print(np.max(tr_image), np.min(tr_image))

    tr_image = table_transform(image, 2.0)
    cv2.imwrite(os.path.join('UNLV/x_1_0101_003.jpg'), tr_image)
    print(np.max(tr_image), np.min(tr_image))

def main():
    i = 0
    images = os.listdir(source)
    total = len(images)
    print('total images = ', total)
    for file_name in images:
        i += 1
        # read the image, transform and write the new image
        image = cv2.imread(os.path.join(source, file_name), 0)   # LOAD AS GRAY SCALE IMAGE
        tr_image = table_transform(image, 0.3)
        cv2.imwrite(os.path.join(destination, file_name), tr_image)
        # verbose
        if i % 10 == 0:
            print(i ,'/', total, ' done')

# test_transform_definition()
#if __name__ == '__main__':
#    main()




