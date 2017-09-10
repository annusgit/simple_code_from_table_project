import os
from PIL import Image 

source = '/home/annus/Desktop/Folders/project/UNLV/UNLV data/transformed'
dest = '/home/annus/Desktop/Folders/project/eval images'

list_ = os.listdir(source)
total = len(list_)
print('total = {}'.format(total))
split = int(0.8*total)
i = 0
for image in list_:
    i += 1
    if i <= split:
        continue

    name, _ = os.path.splitext(image)
    PILimg = Image.open(os.path.join(source, image))
    new_img = os.path.join(dest, name) + '.jpg'
    PILimg.save(new_img)

    # verbose
    print('{} / {} done'.format(i, total))


















