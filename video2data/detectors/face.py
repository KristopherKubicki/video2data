#!/usr/bin/python3 
# coding: utf-8
#

# initialize people detection
print("loading dlib")
detector = dlib.get_frontal_face_detector()
# not allowed for commercial redistribution
# https://github.com/davisking/dlib-models
#  the smaller, 5 segment one is ok.  I might be able to use that
#predictor = dlib.shape_predictor('/home/kristopher/tf_files/scripts/shape_predictor_68_face_landmarks.dat')
# 5 segment seems to work
predictor = dlib.shape_predictor('/home/kristopher/tf_files/scripts/shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('/home/kristopher/tf_files/scripts/dlib_face_recognition_resnet_model_v1.dat')
