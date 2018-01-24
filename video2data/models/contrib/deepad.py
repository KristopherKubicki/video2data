#!/usr/bin/python3
# coding: utf-8

NUM_CLASSES = 16

import sys
import tensorflow as tf
sys.path.append('/home/kristopher/models/research/object_detection')
from utils import label_map_util

# load up the SSD-derived NN I trained on the 16 brand logos
#   this is used for commercial detection.  Average inference is about 0.03s on 1080p

class nnLogo:

  def load(self):
    # now load up the fastest model. Average inference is about 0.03s at 1080p
    print("loading deepad")
    # I should just pull this from google model zoo 
    logo_graph = tf.Graph()
    with logo_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile('models/contrib/deepad/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    self.label_map = label_map_util.load_labelmap('models/contrib/deepad/deepad.pbtxt')
    self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.category_index = label_map_util.create_category_index(self.categories)

    self.image_tensor = logo_graph.get_tensor_by_name('image_tensor:0')
    self.detection_boxes = logo_graph.get_tensor_by_name('detection_boxes:0')
    self.detection_scores = logo_graph.get_tensor_by_name('detection_scores:0')
    self.detection_classes = logo_graph.get_tensor_by_name('detection_classes:0')
    self.num_detections = logo_graph.get_tensor_by_name('num_detections:0')
    self.sess = tf.Session(graph=logo_graph)
    return self

  def test(self,image_np_expanded):
    (self.boxes, self.scores, self.classes, self.num) = self.sess.run(
    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
    feed_dict={self.image_tensor: image_np_expanded})

