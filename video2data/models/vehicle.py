#!/usr/bin/python3
# coding: utf-8
NUM_CLASSES = 2

import sys
import tensorflow as tf
# TODO: tf_slim instead
#sys.path.append('/home/kristopher/models/research/object_detection')
from utils2 import label_map_util

#
# A car / pedestrian flavored model, good for exterior cameras
#  It takes 0.04s for 1080p
#  
# http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2017_11_08.tar.gz
class nnVehicle:
  
  def load(self):
    print("loading vehicle detection")
    kitti_graph = tf.Graph()
    with kitti_graph.as_default():
      kitti_graph_def = tf.GraphDef()
      with tf.gfile.GFile('models/vehicle/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        kitti_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(kitti_graph_def, name='kitti')

    self.klabel_map = label_map_util.load_labelmap('models/vehicle/kitti_label_map.pbtxt')
    self.kcategories = label_map_util.convert_label_map_to_categories(self.klabel_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.kcategory_index = label_map_util.create_category_index(self.kcategories)

    self.kitti_image_tensor = kitti_graph.get_tensor_by_name('kitti/image_tensor:0')
    self.kitti_detection_boxes = kitti_graph.get_tensor_by_name('kitti/detection_boxes:0')
    self.kitti_detection_scores = kitti_graph.get_tensor_by_name('kitti/detection_scores:0')
    self.kitti_detection_classes = kitti_graph.get_tensor_by_name('kitti/detection_classes:0')
    self.kitti_num_detections = kitti_graph.get_tensor_by_name('kitti/num_detections:0')
    self.sess = tf.Session(graph=kitti_graph)
    return self

  def test(self,image_np_expanded):
    (self.kboxes, self.kscores, self.kclasses, self.knum) = self.sess.run(
    [self.kitti_detection_boxes, self.kitti_detection_scores, self.kitti_detection_classes, self.kitti_num_detections],
    feed_dict={self.kitti_image_tensor: image_np_expanded})

