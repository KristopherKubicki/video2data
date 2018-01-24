#!/usr/bin/python3
# coding: utf-8

import sys
import dlib

# https://github.com/davisking/dlib-models

class nnFaces:

  def load(self):
    print('loading facial recognition')
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor('models/faces/shape_predictor_5_face_landmarks.dat')
    self.facerec = facerec = dlib.face_recognition_model_v1('models/faces/dlib_face_recognition_resnet_model_v1.dat')

    return self

  def test(self,image_np):
    self.detector(image_np,1)
