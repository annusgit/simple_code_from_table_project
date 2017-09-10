import os
import xml.etree.ElementTree as et

source = 'Annotations'
dest = 'ground truth text'

xmls = os.listdir(source)
total = len(xmls)
print('total xmls = {}'.format(total))
i = 0

for xml in xmls:

    name, _ = os.path.splitext(xml)
    GT_text_file = os.path.join(dest, name + '.txt')

    tree = et.parse(xml)
    root = tree.getroot()

    objects = root.findall('.//object')

    with open(GT_text_file, 'w') as txt:
        for obj in objects:
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)

            print('{}, {}, {}, {}'.format(xmin, ymin, xmax, ymax), file=txt)




