import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
from transforms import table_transform as trans
import scipy.misc
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# supress the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

PATH_TO_CKPT = os.path.join('check_inference_graph', 'frozen_inference_graph.pb')
PATH_TO_LABELS = '/home/annus/data/my_pascal_label_map.pbtxt'
NUM_CLASSES = 1

def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# load a frozen graph into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# load the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
                                                            label_map, 
                                                            max_num_classes=NUM_CLASSES, 
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# test on all jpg and jpeg files
files_list = os.listdir(os.path.join('object_detection', 'table test images'))
images = [this for this in files_list if (this.endswith('.jpg') or this.endswith('.jpeg'))]
total = len(images)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
dest = os.path.join('object_detection', 'here')

################################### Detection Pipeline ##################################
#1 Resize the image
#2 apply the distance transformation 
#3 pass this image through the network and get the bounding boxes
#4 resize the bounding boxes and draw them on the original image
#########################################################################################
Size = 400, 400
i = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for im_ in images:
            
            i += 1
            image_path = os.path.join('object_detection', 'table test images', im_)
            name, _ = os.path.splitext(im_)

            #1 resize the image
            Original_image = Image.open(image_path)
            Prev_image = Original_image.copy()
            Pwidth, Pheight = Prev_image.size
            Prev_img.thumbnail(Size, Image.ANTIALIAS)
            Nwidth, Nheight = Prev_image.size
            resize_ratio = (Nwidth/Pwidth, Nheight/Pheight)

            #2 apply the distance transformation
            Prev_image = trans(Prev_image)
            resized_image = Prev_image.copy()

            #3 pass through the F-RCNN
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(resized_image)
            original_np = load_image_into_numpy_array(Original_image)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              resize_ratio,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)
            scipy.misc.imsave(os.path.join(dest, name +'.jpg'), image_np)
            print('{} / {}'.format(i, total), ' done')


            # results on the original images
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              resize_ratio,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)
            scipy.misc.imsave(os.path.join(dest, name +'.jpg'), image_np)
            print('{} / {}'.format(i, total), ' done')
