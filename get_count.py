import xml.etree.ElementTree as et
import sys
import os 

directory = '/home/annus/Desktop/Folders/proper_staples_data_set/'+sys.argv[1]
source = [file for file in 
            os.listdir(directory) 
            if file.endswith('.xml')]
    
def getCount():
    total = len(source)
    i = 0
    for xml in source:
        tree = et.parse(os.path.join(directory, xml))
        root = tree.getroot()    
        depth = root.find('size').find('depth').text
        if depth is '3':
            i += 1
    print('{} / {}'.format(i, total))

getCount()

