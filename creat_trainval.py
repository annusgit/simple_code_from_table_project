import os

img_dir = '/home/annus/Desktop/project/UNLV/images'
trainval_txt_file = '/home/annus/Desktop/project/UNLV/annotations/trainval.txt'

im_list = os.listdir(img_dir)
total = len(im_list)
print('total images = ', total)
i = 0

with open(trainval_txt_file, 'w') as txt:
    
    for im in im_list:

        i += 1
        name, _ = os.path.splitext(im)
        print(name + ' 1', file=txt)

        if i % 10 == 0:
            print(i, '/%d written' %total)

        #if i == 4:
        #   break





