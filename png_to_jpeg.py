import os                            # for using directory paths
import cv2
from PIL import Image

# collect directory paths
img_dir = 'UNLV/jpeg transformed images'
dest_dir = 'UNLV/new'
imgs = os.listdir(img_dir)
total = len(imgs)
print('total images = ', total)
size = 400, 400
i = 0

for file_name in imgs:

    i += 1
    name, _ = os.path.splitext(file_name)
    img = Image.open(os.path.join(img_dir, file_name))
    img.thumbnail(size, Image.ANTIALIAS)
    img.save(os.path.join(dest_dir, name) + '.jpg')

    # verbose
    if i % 10 == 0:
        print(i ,'/', total, ' converted')

    #if i == 1: 
    #    break













