#!/usr/bin/python3
# coding: utf-8

NUM_CLASSES = 546

import sys
import tensorflow as tf
#sys.path.append('/home/kristopher/models/research/object_detection')
from utils2 import label_map_util


# Another model that can detect tons of objects. This is currently
# the biggest public model that I know about,  Open Images.  It has 546 objects
#  It takes 0.7s per detection.  12s to load
# 
# http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2017_11_08.tar.gz
#
# the low proposals set is supposed to be twice as fast as the normal one.  i have found that not be the case
# and they run about the same.  

class nnObject:

  def load(self):
    # now load up the fastest model. Average inference is about 0.03s at 1080p
    print("loading open images, this takes a while")
    # I should just pull this from google model zoo 
    oi_graph = tf.Graph()
    with oi_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile('models/objects/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='oi')
    self.olabel_map = label_map_util.load_labelmap('models/objects/oid_bbox_trainable_label_map.pbtxt')
    self.ocategories = label_map_util.convert_label_map_to_categories(self.olabel_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.ocategory_index = label_map_util.create_category_index(self.ocategories)

    self.o_image_tensor = oi_graph.get_tensor_by_name('oi/image_tensor:0')
    self.o_detection_boxes = oi_graph.get_tensor_by_name('oi/detection_boxes:0')
    self.o_detection_scores = oi_graph.get_tensor_by_name('oi/detection_scores:0')
    self.o_detection_classes = oi_graph.get_tensor_by_name('oi/detection_classes:0')
    self.o_num_detections = oi_graph.get_tensor_by_name('oi/num_detections:0')
    self.sess = tf.Session(graph=oi_graph)
    return self

  def test(self,image_np_expanded):
    (self.oboxes, self.oscores, self.oclasses, self.onum) = self.sess.run(
    [self.o_detection_boxes, self.o_detection_scores, self.o_detection_classes, self.o_num_detections],
    feed_dict={self.o_image_tensor: image_np_expanded})

