import tensorflow as tf
import dataset_util
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et

def create_tf_example(example, encoded, dims, corners):
   
    # TODO(user): Populate the following variables from your example.
    width, hieght, channels = dims
    xmin, ymin, xmax, ymax = corners

    filename = example # Filename of the image. Empty if image is not from file
    # encoded_image_data = None # Encoded image bytes
    
    image_format = 'jpeg'.encode('utf8') # b'jpeg' or b'png'

    xmins = [xmin/width] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [xmax/width] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [ymin/hieght] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [ymax/hieght] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = ['table'] # List of string class name of bounding box (1 per box)
    classes = [1] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(hieght),
        'image/width': dataset_util.int64_feature(width),
        # I added this
        #'image/channels': dataset_util.int64_feature(channels),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded.tobytes()),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(str.encode(i) for i in classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(0),
        'image/object/truncated': dataset_util.int64_list_feature(0),
        'image/object/view': dataset_util.bytes_list_feature(i.encode('utf8') for i in ['unspecified']),
    }))
    return tf_example


# define the output file
train_tfrecords_filename = 'tf record transformed images/train_data_jpeg.record'
eval_tfrecords_filename = 'tf record transformed images/eval_data_jpeg.record'
source_dir = 'UNLV/jpeg transformed images'
annotations_dir = 'editted xmls'

train_writer = tf.python_io.TFRecordWriter(train_tfrecords_filename)
eval_writer = tf.python_io.TFRecordWriter(eval_tfrecords_filename)

examples = os.listdir(source_dir)
total = len(os.listdir(source_dir))
print('total images= ', total)
i = 0
split = int(0.8*total)
print('training total = ', split)
print('eval total = ', total - split)

for example in examples:
    i += 1

    # split name and extension
    example_name, _ = os.path.splitext(example)

    # read the image, without any change
    im = cv2.imread(os.path.join(source_dir, example), -1)
    hieght, width, channels = im.shape
    dims = (hieght, width, channels)

    # find the coordinates of b_box from the corresponding xml file
    this_xml = os.path.join(annotations_dir, example_name + '.xml')
    if os.path.isfile(this_xml):
        tree = et.parse(this_xml)
        root = tree.getroot()
        x_min = int(root.find('object').find('bndbox').find('xmin').text)
        y_min = int(root.find('object').find('bndbox').find('ymin').text)
        x_max = int(root.find('object').find('bndbox').find('xmax').text)
        y_max = int(root.find('object').find('bndbox').find('ymax').text)
        corners = (x_min, y_min, x_max, y_max)

    else:
        print(this_xml, ' not found!')

    tf_example = create_tf_example(example=example, encoded=im, dims=dims, corners=corners)

    if i <= split:
        train_writer.write(tf_example.SerializeToString())
        if i % 10 == 0:
            print(i, '/', total, ' done in training set')
    else:
        eval_writer.write(tf_example.SerializeToString())
        if i % 10 == 0:
            print(i, '/', total, ' done in cross validation set')

train_writer.close()
eval_writer.close()













