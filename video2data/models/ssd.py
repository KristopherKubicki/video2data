#!/usr/bin/python3
# coding: utf-8

NUM_CLASSES = 90

import sys
import tensorflow as tf
sys.path.append('/home/kristopher/models/research/object_detection')
from utils import label_map_util

class nnSSD:

  def load(self):
    # now load up the fastest model. Average inference is about 0.03s at 1080p
    print("loading mobilenet")
    # I should just pull this from google model zoo 
    coco_graph = tf.Graph()
    with coco_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile('models/ssd/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='coco')
    self.clabel_map = label_map_util.load_labelmap('models/ssd/mscoco_label_map.pbtxt')
    self.ccategories = label_map_util.convert_label_map_to_categories(self.clabel_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.ccategory_index = label_map_util.create_category_index(self.ccategories)

    self.c_image_tensor = coco_graph.get_tensor_by_name('coco/image_tensor:0')
    self.c_detection_boxes = coco_graph.get_tensor_by_name('coco/detection_boxes:0')
    self.c_detection_scores = coco_graph.get_tensor_by_name('coco/detection_scores:0')
    self.c_detection_classes = coco_graph.get_tensor_by_name('coco/detection_classes:0')
    self.c_num_detections = coco_graph.get_tensor_by_name('coco/num_detections:0')
    self.sess = tf.Session(graph=coco_graph)
    return self

  def test(self,image_np_expanded):
# tmp load
    (self.cboxes, self.cscores, self.cclasses, self.cnum) = self.sess.run(
    [self.c_detection_boxes, self.c_detection_scores, self.c_detection_classes, self.c_num_detections],
    feed_dict={self.c_image_tensor: image_np_expanded})

