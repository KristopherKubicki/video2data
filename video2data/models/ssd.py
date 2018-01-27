#!/usr/bin/python3
# coding: utf-8

NUM_CLASSES = 90

import sys
import tensorflow as tf
import numpy as np
from utils2 import image as v2dimage

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

  def run(self,frame,last_frame=None):
    image_np_expanded = np.expand_dims(frame['rframe'], axis=0)
    (cboxes, cscores, cclasses, cnum) = self.sess.run(
       [self.c_detection_boxes, self.c_detection_scores, self.c_detection_classes, self.c_num_detections],
       feed_dict={self.c_image_tensor: image_np_expanded})
    # TODO: Check to see if the old rectangles are inside the new proposals
    #  if so, increase the weighting
    old_human_rects = []
    if last_frame and last_frame.get('human_rects'):
      old_human_rects = last_frame.get('human_rects')
    frame['human_rects'] = []

    # TODO: check vehicle and ssd rects the same way we check human rects
    frame['vehicle_rects'] = []
    frame['ssd_rects'] = []

    # TODO pythonify
    i = 0
    for cscore in cscores[0]:
    # carry over label and frame from previous detection
      detected_frame = frame['frame']
      detected_label = ''
      if self.ccategory_index[cclasses[0][i]]['name'] in ['person','face']:
        rect_center = v2dimage.rectCenter(v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width'],frame['height']))
        for prect in old_human_rects:
          if prect[0][0] < rect_center[0] < prect[0][2] and prect[0][1] < rect_center[1] < prect[0][3]:
            cscore += 0.2
            if prect[2][:4] == 'CELE':
              detected_label = str(prect[2])
            detected_frame = prect[3]
            break

      w = cboxes[0][i][2] - cboxes[0][i][0]
      h = cboxes[0][i][3] - cboxes[0][i][1]
      if cscore < 0.3:
        break
      if w*h > 0.8:
        continue
      if w*h > 0.6 and cscore < 0.7:
        continue
      #print('\t\tssd',ccategory_index[cclasses[0][i]]['name'],w*h,cscore)

      # TODO: Don't let humans be in the corners of the screen. Detection is at its worst there
      if self.ccategory_index[cclasses[0][i]]['name'] in ['person','face']:
        # enable human tracking here
        # TODO: Don't let human be in the corners of the screen. Detection is at its worst there
        #  don't have to dlib on every frame, especially if we already have a label
        if detected_label is '':
          detected_label = 'Person'
           
          # default, give dlib the smaller face version
          tmpRect = v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width']*frame['scale'],frame['height']*frame['scale'])
          tmp = frame['small'][tmpRect[1]:tmpRect[3],tmpRect[0]:tmpRect[2]:]
          # don't bother with a small version if its a tiny face.  Give dlib the full version
          # TODO: subpixel enhancement for better detection on smaller faces
          if w*h < frame['scale'] or (w*h < 0.3 and frame['frame'] > detected_frame + 3): # SCALER
            tmpRect = v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width'],frame['height'])
            tmp = frame['rframe'][tmpRect[1]:tmpRect[3],tmpRect[0]:tmpRect[2]:]

          # TODO: enable this
          continue
          dets = faces.detector(tmp,1)
          tmp_gray = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
          for k, d in enumerate(dets):
            shape = faces.predictor(tmp_gray,d)
            face_descriptor = faces.facerec.compute_face_descriptor(tmp, shape)
            all_faces.append(face_descriptor)
            labels = dlib.chinese_whispers_clustering(all_faces, 0.5)
            frame['faces'].append(base64.b64encode(np.squeeze(labels[-1])).decode('utf-8')[:-2])
            detected_label = 'CELEB: ' + base64.b64encode(np.squeeze(labels[-1])).decode('utf-8')[:-2]
            cscore += 0.5
            break
          # TODO: No face?  Maybe hurt the human score!
          if len(dets) == 0 and cscore < 0.4:
            cscore -= 0.2
        human_rects.append([v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width'],frame['height']),cscore,detected_label,detected_frame])
        #ok = people_tracker.add(cv2.TrackerKCF_create(),frame['small'],v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width']*SCALER,frame['height']*SCALER))
      elif self.ccategory_index[cclasses[0][i]]['name'] in ['car','bus','truck','vehicle','boat','airplane']:
        if cscore < 0.3:
          continue
        #print('\t\tcar',ccategory_index[cclasses[0][i]]['name'],w*h,cscore)
        frame['vehicle_rects'].append([v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width'],frame['height']),cscore,self.ccategory_index[cclasses[0][i]]['name'],detected_frame])
        #      ok = kitti_tracker.add(cv2.TrackerKCF_create(),frame['gray'],v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width']*SCALER,frame['height']*SCALER))
      elif cscore > 0.7:
        frame['ssd_rects'].append([v2dimage.scaleRect(v2dimage.shiftRect(cboxes[0][i]),frame['width'],frame['height']),cscore,self.ccategory_index[cclasses[0][i]]['name'],detected_frame])
      i += 1

    return frame



  def test(self,image_np_expanded):
# tmp load
    (self.cboxes, self.cscores, self.cclasses, self.cnum) = self.sess.run(
    [self.c_detection_boxes, self.c_detection_scores, self.c_detection_classes, self.c_num_detections],
    feed_dict={self.c_image_tensor: image_np_expanded})

