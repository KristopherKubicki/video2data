#!/usr/bin/python3
# coding: utf-8

NUM_CLASSES = 16

import sys
import tensorflow as tf
import numpy as np
from utils2 import image as v2dimage

#sys.path.append('/home/kristopher/models/research/object_detection')
from utils2 import label_map_util

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
    self.detect_count = 0

    return self

  def run(self,frame,last_frame=None):
   
     if frame['scene_detect'] == frame['frame']:
       self.detect_count = 0

     image_np_expanded = np.expand_dims(frame['rframe'], axis=0)
     (boxes, scores, classes, num) = self.sess.run(
        [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
        feed_dict={self.image_tensor: image_np_expanded})

     # TODO: limit to later trained classes

     i = 0
     for score in scores[0]:
       if score < 0.2:
         continue
        # accept a low score in a corner. but everywhere else it has to be high
        # check size of rect.  If too big, omit it
       w = boxes[0][i][2] - boxes[0][i][0]
       h = boxes[0][i][3] - boxes[0][i][1]

       if classes[0][i] < 12:
         continue

# filter out some stuff 
       if w*h > 0.5 and score < 0.98:
         continue
       if w*h > 0.4 and score < 0.95:
         continue
       if w*h > 0.2 and score < 0.5:
         continue
       if w > 0.7:
         continue

       # check to see if the bounding boxes are in the corners, and substantially increase the weight if detected
       if ((v2dimage.rectInRect(boxes[0][i],(0.8,0.8,1,1)) or v2dimage.rectInRect(boxes[0][i],(0,0.8,0.2,1))) and score > 0.01) or ((v2dimage.rectInRect(boxes[0][i],[0,0,0.2,0.2]) or v2dimage.rectInRect(boxes[0][i],[0.8,0,1,0.2])) and score > 0.1):
         self.detect_count += 1
       if self.detect_count > frame['fps']:
         #print('logo',score,classes[0][i],self.category_index[classes[0][i]]['name'])
         frame['com_detect'] = 'Programming'
         frame['contrib_rects'].append([v2dimage.scaleRect(v2dimage.shiftRect(boxes[0][i]),frame['width'],frame['height']),score,self.category_index[classes[0][i]]['name'],frame['frame']])
         break
       elif score > 0.90:
         #print('logo',score,classes[0][i],self.category_index[classes[0][i]]['name'])
         frame['contrib_rects'].append([v2dimage.scaleRect(v2dimage.shiftRect(boxes[0][i]),frame['width'],frame['height']),score,self.category_index[classes[0][i]]['name'],frame['frame']])
         break
       i += 1
     return frame


  def test(self,image_np_expanded):
    (self.boxes, self.scores, self.classes, self.num) = self.sess.run(
    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
    feed_dict={self.image_tensor: image_np_expanded})

