#!/usr/bin/python3
# coding: utf-8

import sys
import tensorflow as tf
# required to symlink the nets/ path
sys.path.append('/home/kristopher/EAST')
import model as east
import numpy as np
import cv2
import time

# custom module and needs to be refactored
import loadeast


class nnText:

  def load(self):
    print("loading scene text detection")
    self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    self.f_score, self.f_geometry = east.model(self.input_images, is_training=False)
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    saver.restore(self.sess, 'models/text/model.east_icdar2015_resnet_v1_50_rbox.ckpt-49491')

    return self

  def test(self,image_np):
    h = np.size(image_np,0)
    w = np.size(image_np,1)

    max_side_len = 2400
    resize_w = w
    resize_h = h
    if max(resize_h, resize_w) > max_side_len:
      ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
      ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im_resized = cv2.resize(image_np, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
 
    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()
    scores, geometry = self.sess.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [im_resized]})
    timer['net'] = time.time() - start
    boxes, timer = loadeast.detect(score_map=scores, geo_map=geometry,timer=timer)
    text_hulls = []
    if boxes is not None:
      boxes = boxes[:, :8].reshape((-1, 4, 2))
      boxes[:, :, 0] /= ratio_w
      boxes[:, :, 1] /= ratio_h
      for box in boxes:
        if (box[0][1] - box[1][1]) / h < 0.05:
          box[1][1] = box[0][1]
          box[2][1] = box[3][1]
          box[1][0] = box[2][0]
          box[3][0] = box[0][0]

          #loop through to see if we have any neighbors
          # TODO: omit the neighbors
          for box2 in boxes:
            if box2[0][0] > box[1][0] and box2[0][0] - box[1][0] < w * 0.05 and abs(box2[0][1] - box[0][1]) < h * 0.05 and abs(box2[2][1] - box[2][1]) < h * 0.05:
              box[1][0] = box2[0][0]
              box[2][0] = box2[0][0]
        box = np.int0(box)
        hull = cv2.convexHull(box.reshape(-1,1,2))
        text_hulls.append(hull)
     
    # TODO: collapse overlapping hulls
    return text_hulls


