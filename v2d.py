#!/usr/bin/python3 
# coding: utf-8
#
# This is a general framework to fuse a lot of neural network based technologies together.
#  I wanted to monitor my cameras and network TV streams, which are just video / audio feeds, for specific events
#  Most of these events were hard to do with rule based systems. 
#
#  unblink will monitor your video stream for faces, logos, people, text, scene breaks, commercials and advertisements
#    it draws bounding boxes around the objects.  
#
#  The program strings together a bunch of common pipeline techniques to get a sense of what is going on in a feed, in 
#  realtime.  It is prone to underrunning.  If that happens, reduce your resolution
#
# This codebase is highly experimental and for research purposes only
# 
#  It takes a lot of processing power, and is currently single threaded.  
#    My computer:
#     Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz
#  GeForce GTX 1080
#  GeForce GTX 970 (this doesn't help much I think.  But its an extra 4GB of memory)
#
# NVIDIA-SMI 387.34
#
#  usage: ./unblink.py
#    video stream is hardcoded in (need to make it an arg)
#    script will pull up an OpenCV window and play AV from:
#       1.) an HTTP MPEG TS stream (video, audio)
#       2.) an RTSP stream (video only)   
#
#    It will then demux that video into 1080p streams.
#
#    Audio is analyzed for speech, breaks and silence.  
#     One process filters for human speech across audio, and segments it into 1-5 sec bites
#     That is passed to pocketsphinx_continous, which returns a realtime translation. 
#     The translations are to be appended into the eventual output stream
#     
# I want to look through AR goggles and have unblink filter all the ads for me
#   Right now it has a 0-20s delay.  
#
# Originally I used a basic Thread to power the event loop.  It has a way to call a deque, which I used for the audio.
#  when I swithced to multiprocessing, I couldn't peek ahead on the queue, and thus couldnt do audio.  The application
# needs multiprocessing badly!
#import threading
import multiprocessing 


import os, sys, math

import tensorflow as tf

if __name__ == '__main__':
  multiprocessing.set_start_method('spawn')

#
# Consider disabling OpenCL! I've had some weirdness with it
#  (constantly loading, unloading)
#

FPS = 30
WIDTH = 1920
HEIGHT = 1080
WIDTH = 1280
HEIGHT = 720
SAMPLE = 44100

# get ready, you're going to use a lot of this
import numpy as np
from PIL import Image

# and playing with fifo buffers
from fcntl import fcntl, F_GETFL, F_SETFL, ioctl 

import os, sys, math, select
from os import O_RDONLY, O_NONBLOCK, read
import copy
import random

# had to edit the base package for python3 error
# https://bugs.launchpad.net/python-v4l2/+bug/1664158
from v4l2 import *
#import v4l2

# 1.4.1
import tensorflow as tf
from tensorflow.contrib import slim
import datetime
# facial detection
import dlib
import hashlib
import zlib

# 3.4.0
import cv2
cv2.ocl.setUseOpenCL(cv2.ocl.haveOpenCL())
#cv2.ocl.setUseOpenCL(False)

# A few subprocesses open pipes into the 
import subprocess
import time
import re
from queue import Queue
# encode 128D vectors for facial fingerprints
import base64

import video2data.ocr

# I want to set a dict with a key value without initializing the key
from collections import defaultdict

# ## Object detection imports
#  The whole tensorflow models directory is sitting in my home dir
sys.path.append('/home/kristopher/models/research/object_detection')
# I believe I can import these from the "slim" research branch.  Right now it does not
from utils import label_map_util

# this is a pretty good audio playing package.  unblink queues up a 
#  bunch of audio as it comes in from the frames.  The audio needs to be joined
# and preferably on breaks. 
import pygame
pygame.init()
#  44100 samples, 16 bit, mono.  I considered stereo but the speech to text 
# translators all want mono anyway.  We can always add additional channels
pygame.mixer.init(44100, -16, 1)

# initialize the edge distance classifier
hd = cv2.createHausdorffDistanceExtractor()
sd = cv2.createShapeContextDistanceExtractor()

# initialize the tracker
people_tracker = cv2.MultiTracker_create()
oi_tracker = cv2.MultiTracker_create()
logo_tracker = cv2.MultiTracker_create()
face_tracker = cv2.MultiTracker_create()
kitti_tracker = cv2.MultiTracker_create()

image_np = cv2.imread("test.jpg")
#cv2.imshow('Object Detection',image_np)
#cv2.waitKey(1000)
image_np_expanded = np.expand_dims(image_np, axis=0)



# Load up the sources 
sources = [line.rstrip('\n') for line in open('/home/kristopher/tf_files/scripts/sources.txt')]

 
# Initialize text detector
mser = cv2.MSER_create(30,5,600)
mser_scaler = 0.20

if 1==1:
  if 1==1:


    all_faces = []
    while True:
      for source in sources:
        fvs = FileVideoStream(source).load()
        print('source: %s' % source)

        last_gray = None

        prev_scene = {}
        prev_scene['type'] = 'Segment'
        prev_scene['length'] = 0
        prev_scene['caption'] = None
        prev_scene['fingerprint'] = None

        scene_summary = ''
        scene_caption = ''
        shot_summary = ''
        frame_caption = ''
        frame_transcribe = ''
        caption_end = ''
        com_detect = 'Segment'
        logo_detect = ''
        last_caption = ''
        cap_buf = ''

        text_id = 0
        scene_count = 0
        logo_detect_count = 0
        scene_detect = 0
        shot_detect = 0
        audio_detect = 0
        motion_detect = 0
        motion_start = 0
        subtle_detect = 0
        frame_delta = 0
        last_mean = 0
        last_scene = 0
        last_transcribe = 0
        frame = 0
        b_count = 0
        a_count = 0
        break_count = 0
        last_contrast = 0
        last_width = 0
        last_height = 0
        last_shot = 0

        fps_frame = 0
        start_time = time.time()
        channel1 = pygame.mixer.Channel(0)

        logo_rects = []
        oi_rects = []
        kitti_rects = []
        ssd_rects = []
        face_rects = []
        person_rects = []
        text_rects = []
        face_hulls = []
        text_hulls = []
        motion_hulls = []
        motion_cnts = []
        gray_cnts = []
        last_gray_cnts = []
        scene_faces = []
        scene_text = []
        shot_faces = []
        last_frames = []
        last_hist = []

        text_frame = 0
        logo_frame = 0
        oi_frame = 0
        ssd_frame = 0
        kitti_frame = 0
        face_frame = 0

        scene_shotprint = '0b'
        shot_faceprint = '0b'
        shot_voiceprint = '0b'
        shot_videoprint = '0b'
        shot_textprint = '0b'
        shot_audioprint = '0b'

        cast_video = bytearray()

        while fvs.stopped.value == 0:
          if fvs.Q.qsize() < 1:
            print('sleeping',fvs.t.pid)
            time.sleep(0.1)
            continue

          frametime = datetime.datetime.utcnow().utcfromtimestamp(frame / FPS).strftime('%H:%M:%S')
          filetime = datetime.datetime.utcnow().utcfromtimestamp(fvs.microclockbase + fvs.clockbase).strftime('%H:%M:%S')
          local_fps = fps_frame / (time.time() - start_time)
          if caption_end == frametime:
            frame_caption = ''
            caption_end = ''
          if fvs.caption_guide.get(frametime):
            frame_caption = fvs.caption_guide[frametime]['caption']
            caption_end = fvs.caption_guide[frametime]['stop']
            if frame_caption is not last_caption:
              # sometimes captions get duplicated (truncated, appended to the previous caption).  It's a function of the transcriber.  
              #  see if we can filter those out
              scene_caption += frame_caption
              scene_summary += shot_summary
              last_caption = frame_caption
          
          start = time.time()

          #print("output_a",time.time() - start)
          waitkey = smoothWait()
          fvs.ff = False

          print("output_b",time.time() - start,fvs.Q.qsize())
          data = fvs.read()
          print("output_c",time.time() - start,fvs.Q.qsize())
          image_np = data['frame']
          original_frame = image_np

          if data.get('transcribe'):
            frame_transcribe = data['transcribe']
            last_transcribe = frame


          frame += 1
          if frame % 60 == 0:
            start_time = time.time()
            fps_frame = 0
          fps_frame += 1   

 
#
# Motion and Shot Detection. 
#  Creates a few masks that are used for underlying detectors.  Don't run NNs if there is no need...
#
          frame_mean = data['frame_mean']
          # TODO: detect disolves (contrast delta)
          #  blurs are also a way to get at that
          shot_type = ''
          last_frames.append(frame_mean)
          if last_gray is not None and data['gray'] is not None and last_gray.shape == data['gray'].shape:
            tstart = time.time()
            h1,w1 = last_gray.shape
            if time.time() - tstart > 0.01:
              print("output_080a",time.time() - tstart,h1,w1)

            tstart = time.time()
            frame_delta = cv2.absdiff(last_gray, data['gray'])
            if time.time() - tstart > 0.01:
              print("output_081a",time.time() - tstart,h1,w1,len(last_gray),len(data['gray']),len(frame_delta))

            tstart = time.time()
            frame_delta = cv2.blur(frame_delta,(10,10))
            if time.time() - tstart > 0.01:
              print("output_081b",time.time() - tstart,h1,w1)

            tstart = time.time()
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            if time.time() - tstart > 0.01:
              print("output_082",time.time() - tstart)

            tstart = time.time()
            motion_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
            if time.time() - tstart > 0.01:
              print("output_083a",time.time() - tstart,len(thresh))
            motion_hulls = [cv2.convexHull(p.reshape(-1, 1, 2) * int(1/mser_scaler)) for p in motion_cnts]

            if time.time() - tstart > 0.01:
              print("output_083",time.time() - tstart)

            gray_count = cv2.countNonZero(data['gray'])
            gray_percent = gray_count / (data['width'] * data['height'] * mser_scaler * mser_scaler)
            gray_motion_mean = np.sum(thresh) / float(thresh.shape[0] * thresh.shape[1])

            if time.time() - tstart > 0.01:
              print("output_084",time.time() - tstart)
              #cv2.imshow('motion',thresh)


            # Gradual fade in and out checker
            if(len(last_frames) > 5):
              last_frames.pop(0)

            edges = cv2.Canny(data['gray'],100,200)
            _, gray_cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_TC89_KCOS)
            edge_delta_est = 0
            #if last_gray_cnts is not None and len(last_gray_cnts) > 0 and len(gray_cnts) > 0 and abs(len(gray_cnts[0]) - len(last_gray_cnts[0]))/len(last_gray_cnts[0]) > 2:
            #  edge_delta_est = hd.computeDistance(gray_cnts[0],last_gray_cnts[0])
            # print(len(gray_cnts[0]),np.shape(gray_cnts[0])[0],np.shape(last_gray_cnts[0])[0],d1)

            last_motion = motion_detect
            if frame_mean < 5 or frame_mean > 250 and len(edges) == 0:
              shot_type = 'Blank'
              last_shot = shot_detect
              shot_detect = frame
            elif last_width != data['width'] or last_height != data['height']:
              shot_type = 'Resize'
              last_shot = shot_detect
              shot_detect = frame
            elif abs(frame_mean - last_mean) > 5 and chi2_distance(data['hist'],last_hist) > 1:
              shot_detect = frame
              last_shot = shot_detect
              shot_type = 'Hist'
            elif abs(last_contrast - data['contrast']) > 2 and abs(frame_mean - last_mean) > 0.5:
              shot_detect = frame
              last_shot = shot_detect
              shot_type = 'Contrast'
            elif abs(frame_mean - last_mean) > 2 and chi2_distance(data['hist'],last_hist) > 0.5 and frame > shot_detect + 5 and gray_motion_mean > 5:
              shot_type = 'Motion'
              last_shot = shot_detect
              shot_detect = frame
              motion_detect = frame
            elif len(motion_hulls) > 0:
              motion_detect = frame

            if motion_detect and last_motion != frame - 1:
              motion_start = motion_detect
            if shot_detect == frame:
              motion_detect = frame

          if shot_detect == frame:
            break_count += 1
          else:
            break_count = 0
          last_hist = data['hist']
          last_mean = data['frame_mean']
          last_gray = data['gray']
          last_width = data['width']
          last_height = data['height']
          last_contrast = data['contrast']
          last_gray_cnts = gray_cnts
          #print("output_09b",time.time() - start)

          # audio has to break in order to trigger a scene change
          # potentially, I should only run this when the shot detect > frame - FPS
          audio_type = ''
          if data.get('audio'):
            scene_length = '%d' % int((frame - last_scene + break_count) / FPS)
            if data['audio_level'] == 0:
              # don't do this if there are contours on the screen!
              if len(edges) == 0 or scene_length in [5,10,15,30,45,60,90,120]:
                audio_type = 'Hard'
                audio_detect = frame
                #print('\t%s\tHard Audio Break ' % shot_type,scene_length,frame,break_count,'%.2f' % data['audio_level'],data['silence'])
              else:
                print('\t%s\tFalse Audio Break ' % shot_type,scene_length,frame,break_count,'%.2f' % data['audio_level'],data['silence'])


            elif data['audio_level'] < 10 and scene_length in [5,10,15,30,45,60,90,120]:
              audio_type = 'Soft'
              audio_detect = frame
              #print('\t%s\tSoft Audio Break ' % shot_type,scene_length,break_count,frame,'%.2f' % data['audio_level'],data['silence'])
            elif data['audio_level'] < 20 and data['silence'] > 5:
              audio_type = 'Gray'
              #print('\t%s\tGray Audio Break ' % shot_type,scene_length,frame,break_count,'%.2f' % data['audio_level'],data['silence'])
              audio_detect = frame
            elif data['audio_level'] < 30 and data['silence'] > 10:
              audio_type = 'Micro'
              #print('\t%s\tMicro Audio Break ' % shot_type,scene_length,frame,break_count,'%.2f' % data['audio_level'],data['silence'])
              audio_detect = frame

########
#
#  SCENE BREAK DETECTOR
#
# reset metrics if we detect scene break
#
          frame_type = '%s%s%d' % (audio_type,shot_type,break_count)
          if shot_detect == frame and (frame-last_shot + break_count > FPS / 2):
            print('\t[%s] Title: Unamed Shot %d' % (filetime,frame))
            print('\t[%s] Length: %.02fs %d frames' % (filetime,float((frame - last_shot + break_count) / FPS),frame-last_shot + break_count))
            print('\t[%s] Detect Method: %s ' % (filetime,frame_type))
            print('\t[%s] Signature: %s' % (filetime,phash_bits(shot_audioprint)))  
            if shot_summary:
              print('\t[%s] Description: %s' % (filetime,shot_summary.title()))
            if prev_scene['caption']:
              # TODO: this should really be shot_caption. frame needs to roll up to shot
              print('\t[%s] Caption: %s' % (filetime,frame_caption))
            if shot_faces:
              print('\t[%s] People: %s' % (filetime,set(shot_faces)))
            if text_rects:
              print('\t[%s] Letters: %s' % (filetime,len(text_rects)))

# give a little bit of jitter  
          if audio_detect == frame and frame < shot_detect + 5:
            last_scene = scene_detect 
            scene_detect = frame
            # TODO: make this based on break_count instead
            if (frame - last_scene) / FPS > 1:
              prev_scene = {}
              prev_scene['length'] = float((frame - last_scene + break_count) / FPS)
              if com_detect != 'Programming' and int(prev_scene['length']) in [10,15,30,45,60]:
                com_detect = 'Commercial'
              prev_scene['caption'] = scene_caption
              prev_scene['fingerprint'] = phash_bits(scene_shotprint)
              prev_scene['type'] = com_detect
              prev_scene['detect'] = frame_type

              print('[%s] Title: Unnamed %s' % (filetime,com_detect))
              #print('[%s] Source: %s' % (filetime,fvs.stream))  
              print('[%s] Length: %.02fs' % (filetime,prev_scene['length']))  
              #print('[%s] Type: %s ' % (filetime,com_detect))
              print('[%s] Detect Method: %s ' % (filetime,prev_scene['detect']))  
              print('[%s] Signature: %s' % (filetime,prev_scene['fingerprint']))  
              if shot_summary:
                print('[%s] Description: %s' % (filetime,shot_summary.title()))
              if prev_scene['caption']:
                print('[%s] Caption: %s' % (filetime,prev_scene['caption']))
              if scene_faces:
                print('[%s] People: %s' % (filetime,set(scene_faces)))
              if scene_text:
                print('[%s] Letter Shots: %s' % (filetime,len(scene_text)))

              caption_end = ''
              frame_caption = ''
              frame_transcribe = ''
              scene_caption = ''
              logo_detect = ''
              logo_detect_count = 0
              scene_count += 1
              scene_faces = []
              com_detect = 'Segment'
            scene_shotprint = '0b'
          #print("output_10",time.time() - start)
########

##################
#
# SHOT DETECTOR
#
# if we detect this is a new shot, then all of the detection needs to be redone
          #print("output_0",time.time() - start)
          if shot_detect == frame:
            scene_text.append(text_hulls)
            shot_summary = ''
            face_rects = []
            person_rects = []
            text_rects = []
            face_hulls = []
            text_hulls = []
            ssd_rects = []
            oi_rects = []
            kitti_rects = []
            logo_rects = []
            shot_faces = []
            motion_cnts = []
            motion_hull = []
            oi_tracker = None
            kitti_tracker = None
            logo_tracker = None
            face_tracker = None

            # drain queues on a shot change
            while fvs.ocr.input.qsize() > 0:
               fvs.ocr.input.get()
            while fvs.ocr.output.qsize() > 0:
               fvs.ocr.output.get()

            #fvs.ocr.input = multiprocessing.Queue(maxsize=1024)
            #fvs.ocr.output = multiprocessing.Queue(maxsize=1024)

            shot_audioprint = '0b'
            shot_textprint = '0b'
            shot_voiceprint = '0b'
            shot_videoprint = '0b'
            shot_faceprint = '0b'
            scene_shotprint += '1'
            #print("shot detect!",frame,shot_detect,motion_detect,subtle_detect,last_frames)
          else:
            scene_shotprint += '0'


################
#
# Audio fingerprint.  This is where we take a segment of a clip, and map the 
#  fingerprint.  Check the fingerprint against databases (dejavu) to see if 
#  this clip exists already.  Can be pretty low quality fingerprint, as long as it is 
#  fast.  Also, it's OK to wait a few seconds.  Longer segments take less storage to fingerprint
# 
#  (also, this part enqueues the audio to play)
# 
          if False and fvs.Q.qsize() > 40 and frame % 30 == 0:
            play_audio = bytearray()
            for i in range(10,40):
              if fvs.Q.queue[i].get('audio'):
                play_audio += pygame.mixer.Sound(fvs.Q.queue[i]['audio'])
            if play_audio is not None:    
              sound1 = pygame.mixer.Sound(play_audio)
              channel1.queue(sound1)

###################
#
#  Letterbox detector.  This lets us know if the box size has changed, but also if detected, send a smaller
#    crop to the image processors

###############
#
# MSER - this is a high-contrast filter that looks for text.  
#  Text is the highest indicator of advertising. Detects individual letters
#  Runs a neural network underneath
# 
          # loop through things that have come back from the ocr subprocess
          tstart = time.time()
          while fvs.ocr.output.qsize() > 0:
            local_id,text,score = fvs.ocr.output.get()
            i = 0
            for rect in text_rects:
              if local_id == rect[4] and int(score) > text_rects[i][2]:
                text_rects[i][1] = text
                text_rects[i][2] = int(score)
                break
              i += 1
            if time.time() - tstart > 0.1:
              print('ocr1b',time.time() - tstart,local_id)

          if (frame == shot_detect+7 or frame == shot_detect+17 or frame == shot_detect + 33) or fvs.image or frame % 71 == 0:
            text_frame = frame
            start = time.time()
            # if its close to horizontal, don't rotate it.  just extract with some extra
            # consider adding extras 
            old_text_rects = text_rects
            text_rects = []
            if True:
              text_hulls = []
              im_resized, (ratio_h,ratio_w) = resize_image(data['frame'])
              timer = {'net': 0, 'restore': 0, 'nms': 0}
              start = time.time()
              # text recognition neural network
              scores, geometry = ocr_sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
              timer['net'] = time.time() - start
              boxes, timer = loadeast.detect(score_map=scores, geo_map=geometry,timer=timer)
              if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

                # only do this if its already horizontal? 
                # less motion tracking on horizontal text probably
                #  some processed in realtime, some just for records
                # if there is lots of boxes, just do some
                tmp_pipes = []
                for box in boxes:
                   tstart = time.time()
                   if (box[0][1] - box[1][1]) / data['height'] < 0.05:
                     box[1][1] = box[0][1]
                     box[2][1] = box[3][1]
                     box[1][0] = box[2][0]
                     box[3][0] = box[0][0]

                     # loop through to see if we have any neighbors
                     # group for accuracy improvements
                     # TODO: omit the neighbors
                     for box2 in boxes:
                        if box2[0][0] > box[1][0] and box2[0][0] - box[1][0] < data['width'] * 0.05 and abs(box2[0][1] - box[0][1]) < data['height'] * 0.05 and abs(box2[2][1] - box[2][1]) < data['height'] * 0.05:
                          box[1][0] = box2[0][0]
                          box[2][0] = box2[0][0]
                   box = np.int0(box)
                   hull = cv2.convexHull(box.reshape(-1, 1, 2))
                   text_hulls.append(hull)

                   (x,y,w,h) = cv2.boundingRect(box)
                   if w < data['width'] * 0.01:
                     print('skipping small text',w,data['width'])
                     continue
                   if h < data['height'] * 0.01:
                     print('skipping small text',h,data['height'])
                     continue

                   detected_id = text_id
                   detected_label = '?'
                   detected_frame = frame
                   detected_confidence = 0
                   rect_center = cntCenter(box)

                   for trect in old_text_rects:
                     if trect[0][0] < rect_center[0] < trect[0][2] and trect[0][1] < rect_center[1] < trect[0][3]:
                       detected_confidence = trect[2]
                       detected_label = str(trect[1])
                       detected_frame = trect[3]
                       detected_id = trect[4]
                       # consider reading from subprocess here
                       break
                   # don't OCR every frame (unless this is an image)
                   if fvs.image or detected_confidence < 90:
                     lstart = time.time()
                     tmp = hullHorizontalImage(data['frame'],box)
                     #fvs.ocr.input.put([text_id,tmp.copy(),detected_label,detected_confidence])
                     text_id += 1
                     if time.time() - tstart > 0.01:
                       print('warning lstart',time.time() - tstart)
                   #print('\t\tOCR:',detected_label,detected_confidence)
                   text_rects.append([[box[0][0],box[0][1],box[2][0],box[2][1]],detected_label,detected_confidence,detected_frame,detected_id])
            if time.time() - start > 0.1:
              print("warning ocr_done",len(text_hulls),len(text_rects),time.time() - start)
          #print("output_1",time.time() - start)

##########
# High FPS object detector. Look for stuff that will inform other neural networks
#  Do you see people?  OK, then we can run Dlib fingerprint
#
# just do this once per shot, or every 30 seconds or so
#  doesn't have to be all the time
# mobilenet, then dlib if we see people
#
          if (frame == shot_detect + 3 or fvs.image or (frame <= motion_start + 4 and frame % 5 == 0) or frame % 31 == 0):
            ssd_frame = frame
            #print('\t\tssd detector',frame,local_fps)
            tstart = time.time()
            # 0.03s
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (cboxes, cscores, cclasses, cnum) = coco_sess.run(
                [c_detection_boxes, c_detection_scores, c_detection_classes, c_num_detections],
                feed_dict={c_image_tensor: image_np_expanded})
            # TODO: Check to see if the old rectangles are inside the new proposals
            #  if so, increase the weighting
            old_person_rects = person_rects
            ssd_rects = []
            kitti_rects = []
            person_rects = []

            #  It doesn't really make sense to run this
            #  the SSD module is generally faster than the motion tracker
            people_tracker = cv2.MultiTracker_create()
            kitti_tracker = cv2.MultiTracker_create()

            i = 0
            for cscore in cscores[0]:
              # carry over label and frame from previous detection
              detected_frame = frame
              detected_label = ''
              if ccategory_index[cclasses[0][i]]['name'] == 'person':
                rect_center = rectCenter(scaleRect(shiftRect(cboxes[0][i]),data['width'],data['height']))
                for prect in old_person_rects:
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

              # TODO: Don't let people be in the corners of the screen. Detection is at its worst there
              if ccategory_index[cclasses[0][i]]['name'] == 'person':
                #  don't have to dlib on every frame, especially if we already have a label
                if detected_label is '':
                  detected_label = 'Person'
                  tstart = time.time()
                  # TODO: use the full image instead of the shrunk version. Consider enlarge on top portion
                  #  and probably only take the top half
                  tmpRect = scaleRect(shiftRect(cboxes[0][i]),data['width']*mser_scaler,data['height']*mser_scaler)
                  tmp = data['small'][tmpRect[1]:tmpRect[3],tmpRect[0]:tmpRect[2]:]
                  if w*h < 0.1 or (w*h < 0.3 and frame > detected_frame + 3 and frame > motion_detect +3 and frame % 3 == 0): # mser_scaler
                    tmpRect = scaleRect(shiftRect(cboxes[0][i]),data['width'],data['height'])
                    tmp = data['frame'][tmpRect[1]:tmpRect[3],tmpRect[0]:tmpRect[2]:]

                  dets = detector(tmp,1)
                  if time.time() - tstart > 0.15:
                    print("warn 2dlib_done0",k,time.time() - start)
              
                  # TODO: subpixel enhancement for better detection
                  tmp_gray = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
                  for k, d in enumerate(dets):
                    shape = predictor(tmp_gray,d)
                    face_descriptor = facerec.compute_face_descriptor(tmp, shape)
                    all_faces.append(face_descriptor)
                    if time.time() - tstart > 0.15:
                      print("warn 2dlib_done1",k,time.time() - start)
                    if time.time() - tstart > 0.15:
                      print("warn 2dlib_done2",k,time.time() - start)
                    labels = dlib.chinese_whispers_clustering(all_faces, 0.5)
                    scene_faces.append(base64.b64encode(np.squeeze(labels[-1])).decode('utf-8')[:-2])
                    shot_faces.append(base64.b64encode(np.squeeze(labels[-1])).decode('utf-8')[:-2])
                    detected_label = 'CELEB: ' + base64.b64encode(np.squeeze(labels[-1])).decode('utf-8')[:-2]
                    cscore += 0.5
                    break
                  # TODO: No face?  Maybe hurt the person score!
                  if len(dets) == 0 and cscore < 0.4:
                    cscore -= 0.2
                    continue
                person_rects.append([scaleRect(shiftRect(cboxes[0][i]),data['width'],data['height']),cscore,detected_label,detected_frame])
                ok = people_tracker.add(cv2.TrackerKCF_create(),data['small'],scaleRect(shiftRect(cboxes[0][i]),data['width']*mser_scaler,data['height']*mser_scaler))
              elif ccategory_index[cclasses[0][i]]['name'] in ['car','bus','truck','vehicle','boat','airplane']:
                if cscore < 0.3:
                  continue
                #print('\t\tcar',ccategory_index[cclasses[0][i]]['name'],w*h,cscore)
                kitti_rects.append([scaleRect(shiftRect(cboxes[0][i]),data['width'],data['height']),cscore,ccategory_index[cclasses[0][i]]['name'],frame])
                ok = kitti_tracker.add(cv2.TrackerKCF_create(),data['gray'],scaleRect(shiftRect(cboxes[0][i]),data['width']*mser_scaler,data['height']*mser_scaler))
              elif cscore > 0.6:
                ssd_rects.append([scaleRect(shiftRect(cboxes[0][i]),data['width'],data['height']),cscore,ccategory_index[cclasses[0][i]]['name'],frame])
              i += 1

            if time.time() - tstart > 0.1:
              print("warning coco_done",time.time() - tstart)

########
# High FPS Logo detector. This is one of our proprietary detectors.  Logos help us spot commercials
#  workers, endorsements, and lots of other things.  Ours is trained on 16 classes including 
#  dental care, alcohol, and network brands.
#
          if (frame == shot_detect+3 or (frame >= motion_start + 10 and frame < motion_start + 10 and frame % 10 == 0) or (frame > subtle_detect and frame < subtle_detect + 30 and frame % 10 == 0) or frame % 601 == 0):
            #print('\t\tlogo detector')
            logo_frame = frame
            start = time.time()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = logo_sess.run(
                [da_detection_boxes, da_detection_scores, da_detection_classes, da_num_detections],
                feed_dict={da_image_tensor: image_np_expanded})
            logo_rects = []
            logo_tracker = cv2.MultiTracker_create()

            i = 0
            for score in scores[0]:
              if score < 0.2:
                continue
              # accept a low score in a corner. but everywhere else it has to be high
              # check size of rect.  If too big, omit it
              w = boxes[0][i][2] - boxes[0][i][0]
              h = boxes[0][i][3] - boxes[0][i][1]

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
              if ((rectInRect(boxes[0][i],(0.8,0.8,1,1)) or rectInRect(boxes[0][i],(0,0.8,0.2,1))) and score > 0.01) or ((rectInRect(boxes[0][i],[0,0,0.2,0.2]) or rectInRect(boxes[0][i],[0.8,0,1,0.2])) and score > 0.1):
                logo_detect_count += 1
                if logo_detect_count > FPS:
                  com_detect = 'Programming'
                logo_rects.append([scaleRect(shiftRect(boxes[0][i]),data['width'],data['height']),score,category_index[classes[0][i]]['name'],frame])
                break
                  
              elif score > 0.95:
                logo_rects.append([scaleRect(shiftRect(boxes[0][i]),data['width'],data['height']),score,category_index[classes[0][i]]['name'],frame])
                break
              i += 1
              if time.time() - start > 0.1:
                print("warning logo_done",time.time() - start)
                break

          # if its been a long time, and no scene break, it's probably programming
          if frame - scene_detect > 240 * FPS and com_detect is '':
              com_detect = 'Long Programming'
          #print("output_4",time.time() - start)


###############
#
# FasterRCNN Slow detector. (0.8s) Using Open Images, tell if what is on the scene is one of 546 items. 
#
#   Only run on "deep" scenes.  Where the shot hasn't changed in a long time
# 
#  WARNING: This takes TWELVE SECONDS (12) to run the first time on my computer.  
# 
          #if oi_sess is not None and (fvs.image or (frame <= shot_detect + 67 and frame % 67 == 0) or frame % 173 == 0):
          #if oi_sess is not None and (fvs.image or (frame <= shot_detect + 32 and frame % 33 == 0) or (frame > motion_detect + 173 and frame % 173 == 0)):
          if oi_sess is not None and (fvs.image or (frame <= scene_detect + 32 and frame % 33 == 0) or (frame <= shot_detect + 273 and frame % 273 == 0) or (frame > motion_detect + 373 and frame % 373 == 0)):
            #print('\t\toi detector',frame,local_fps)
            oi_frame = frame
            tstart = time.time()
            # ive seen this take 12s the first time it loads, then its fast
            #  usually 0.8s otherwise
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (oboxes, oscores, oclasses, onum) = oi_sess.run(
                [oi_detection_boxes, oi_detection_scores, oi_detection_classes, oi_num_detections],
                feed_dict={oi_image_tensor: image_np_expanded})
            oi_rects = []
            oi_tracker = cv2.MultiTracker_create()

            i = 0
            for oscore in oscores[0]:
              if oscore < 0.01:
                break
              w = oboxes[0][i][2] - oboxes[0][i][0]
              h = oboxes[0][i][3] - oboxes[0][i][1]
              if w*h > 0.8:
                continue
              if w*h > 0.6 and oscore < 0.7:
                continue
             # print('\t\toi',ocategory_index[oclasses[0][i]]['name'],w*h,oscore)

              # only note people if you're really sure
              # TODO: dlib on face? face_rects?
              # Maybe only draw these if they are inside a person
              #if ocategory_index[oclasses[0][i]]['name'] in ['Face','Nose','Eye','Mouth','Arm','Clothing','Hat','Tie']:

              if ocategory_index[oclasses[0][i]]['name'] in ['Person','Man','Woman','Boy','Girl','Face']:
                if oscore < 0.3:
                  continue
                # overwrite the person label with the ID 
                person_rects.append([scaleRect(shiftRect(oboxes[0][i]),data['width'],data['height']),oscore,ocategory_index[oclasses[0][i]]['name'].lower(),frame])
                # hopefully people_tracker exists?
                if people_tracker:
                  ok = people_tracker.add(cv2.TrackerKCF_create(),data['gray'],scaleRect(shiftRect(oboxes[0][i]),data['width']*mser_scaler,data['height']*mser_scaler))
              elif ocategory_index[oclasses[0][i]]['name'] in ['Car','Bus','Truck','Boat','Airplane','Vehicle','Tank','Vehicle','Van','Land vehicle','Limousine','Watercraft','Barge']:
                if oscore < 0.2:
                  continue
                # TODO: this only persists as long as the ssd tracker doens't overwrite it. It does help with the weighting on the next ssd pass
                kitti_rects.append([scaleRect(shiftRect(oboxes[0][i]),data['width'],data['height']),oscore,ocategory_index[oclasses[0][i]]['name'].lower(),frame])
                ok = kitti_tracker.add(cv2.TrackerKCF_create(),data['gray'],scaleRect(shiftRect(oboxes[0][i]),data['width']*mser_scaler,data['height']*mser_scaler))
              elif len(oi_rects) < 5:
                oi_rects.append([scaleRect(shiftRect(oboxes[0][i]),data['width'],data['height']),oscore,ocategory_index[oclasses[0][i]]['name'].lower(),frame])
                ok = oi_tracker.add(cv2.TrackerKCF_create(),data['small'],scaleRect(shiftRect(oboxes[0][i]),data['width']*mser_scaler,data['height']*mser_scaler))
              i += 1
            if time.time() - tstart > 1:
              print("warning oi_done",time.time() - start)

###################33
#   Scene Describer
#   Takes the image and turns it into text.  This can be used to trigger other detectors or vice versa
#    Takes 0.4s 
          if len(oi_rects) > 0 and (frame == shot_detect+37 or fvs.image or (frame > shot_detect + 173 and frame % 173 == 0)):
            tstart = time.time()
            image_cap = cv2.imencode('.jpg', image_np)[1].tostring()

            include_labels = []
            exclude_labels = []
            # keep gender neutral and low
            exclude_labels.extend(['men','women','boys','girls','child','guy','girl','boy','man','woman','skateboarder'])
            if len(person_rects) == 0 or person_rects[0][1] < 0.5:
              exclude_labels.extend(['person','people'])
            elif len(person_rects) == 1 and person_rects[0][1] > 0.6:
              include_labels.extend(['person'])
            elif len(person_rects) > 1 and person_rects[0][1] > 0.5 and person_rects[1][1] > 0.5:
              include_labels.extend(['people'])
            else:
              exclude_labels.extend(['person','people'])

            # TODO: do this with the whole COCO set
            if 'toilet' not in include_labels:
              exclude_labels.append('toilet')
            if 'wii' not in include_labels:
              exclude_labels.append('wii')
            if 'cell phone' not in include_labels:
              exclude_labels.extend(['cell','phone','cellphone'])
            if 'scissors' not in include_labels:
              exclude_labels.append('scissors')
            if 'parking meter' not in include_labels:
              exclude_labels.extend(['parking','meter'])
            if 'fire hydrant' not in include_labels:
              exclude_labels.extend(['fire','hydrant'])
            if 'boat' not in include_labels:
              exclude_labels.append('boat')
              
              
            if len(kitti_rects) > 0:
              for i, rect in enumerate(kitti_rects):
                include_labels.extend(rect[2].split())

            if len(oi_rects) > 0:
              for i, rect in enumerate(oi_rects):
                include_labels.extend(rect[2].split())
            #    print('oi_rect',i,rect[1],rect[2])
            else:
              for i, rect in enumerate(ssd_rects):
                include_labels.extend(rect[2].split())
            #    print('ssd_rect',i,rect[1],rect[2])

            appearances = defaultdict(int)
            for curr in include_labels:
              appearances[curr] += 1
            for curr in include_labels:
              # naive nlp right now
              if appearances[curr] > 1 and curr[:-1] is not 's':
                include_labels.extend(['%ss' % curr])

            # Note: I implemented modifications to the beam search
            # TODO: remove captions that are repeats (boring, not interesting: a hat is next to a hat)
            captions = generator.beam_search(imcap_sess, image_cap,include_labels,exclude_labels)
            captions.sort(key=lambda x: x.score)
            shot_summary = ''
            for i, caption in enumerate(captions):
              sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
              tmp_shot_summary = "%s : %f " % (' '.join(sentence).replace(' .','.').replace(' \'s','\'s'),math.exp(caption.logprob))
              #print("\t\t> %s" % tmp_shot_summary)
            for i, caption in enumerate(captions):
              sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
              shot_summary = ' '.join(sentence).replace(' .','.').replace(' \'s','\'s')
              #print("\t\t< %s" % shot_summary)
              break
            if time.time() - tstart > 1:
              print("warning imcap_done",time.time() - tstart)

###########
# Motion tracking.  Do this instead of detecting an object
#  TODO: Break this up 
#
          if waitkey > 8:
            # we only need to motion detect if an object we detect is inside a motion contour. It is easy to do that, so we might as well
            # TODO: Right now I am only tracking inside a person.  I should track near
            # note: don't run if there are a lot of people, it gets computationally intense
            if people_tracker is not None and person_rects is not None and len(person_rects) > 0 and frame > ssd_frame and frame % len(person_rects) == 0 and len(person_rects) < 5:
              encapsulated = False
              for cnt in motion_cnts: 
                 rect_center = cntCenter(cnt)
                 for person_rect in person_rects:
                   w = person_rect[0][2] - person_rect[0][0]
                   h = person_rect[0][3] - person_rect[0][1]
                   # TODO: sometimes it tracks too much.  Slow it down if the person is just idling in place
                   if w*h/(data['width']*data['height']) < 0.3 and pointInRect(rect_center,scaleRect(person_rect[0],mser_scaler,mser_scaler)) and len(person_rects) < 2:
                     encapsulated = True
                     break
                   if w*h/(data['width']*data['height']) < 0.4 and pointInRect(rect_center,scaleRect(person_rect[0],mser_scaler,mser_scaler)) and len(person_rects) < 3:
                     encapsulated = True
                     break



              if encapsulated:
                tstart = time.time()
                bboxes = people_tracker.update(data['small'])[1]
                if time.time() - tstart > 0.1:
                  print("warning people_tracker_done",time.time() - tstart,frame,local_fps)
                good_rects = 0
                i = 0

                for bbox in bboxes:
                  person_rects[i][0] = scaleRect(bbox,1/mser_scaler,1/mser_scaler)
                  if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                    good_rects += 1
                  i += 1
                if good_rects == 0:
                  people_tracker = None
                  person_rects = []

            # this makes the most sense to track
            #  put cars in separate group, as they are high speed
            if kitti_rects is not None and len(kitti_rects) > 0 and frame > ssd_frame:
              encapsulated = False
              for cnt in motion_cnts: 
                 rect_center = cntCenter(cnt)
                 for car_rect in kitti_rects:
                   w = car_rect[0][2] - car_rect[0][0]
                   h = car_rect[0][3] - car_rect[0][1]
                   if w*h/(data['width']*data['height']) < 0.3 and pointInRect(rect_center,scaleRect(car_rect[0],mser_scaler,mser_scaler)):
                     encapsulated = True
                     break

              if encapsulated:
                tstart = time.time()
                bboxes = kitti_tracker.update(data['small'])[1]
                if time.time() - tstart > 0.1:
                  print("warning kitti_tracker_done",time.time() - tstart,local_fps)
                good_rects = 0
                i = 0
                for bbox in bboxes:
                  kitti_rects[i][0] = scaleRect(bbox,1/mser_scaler,1/mser_scaler)
                  if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                    good_rects += 1
                  i += 1
                if good_rects == 0:
                  kitti_tracker = None
                  kitti_rects = []

            if oi_frame != frame and oi_tracker is not None and oi_rects is not None and len(oi_rects) > 0 and frame % len(oi_rects) == 0:
              encapsulated = False
              for cnt in motion_cnts: 
                 rect_center = cntCenter(cnt)
                 for car_rect in kitti_rects:
                   if pointInRect(rect_center,car_rect[0]):
                     encapsulated = True
                     break

              if encapsulated:
                tstart = time.time()
                bboxes = oi_tracker.update(data['small'])[1]
                if time.time() - tstart > 0.2:
                  print("warning oi_tracker_done",time.time() - tstart)
                good_rects = 0
                i = 0
                for bbox in bboxes:
                  oi_rects[i][0] = scaleRect(bbox,1/mser_scaler,1/mser_scaler)
                  if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                    good_rects += 1
                  i += 1
                if good_rects == 0:
                  oi_tracker = None
                  oi_rects = []

            if face_frame != frame and face_tracker is not None and face_rects is not None and len(face_rects) > 0:
            #if 2==1:
              # check if motion is fully enclosed in a face
              encapsulated = True
              for cnt in motion_cnts: 
                 rect2 = cv2.boundingRect(cnt)
                 outside_people = False
                 for rect1 in face_rects:
                   if rectInRect(rect2,rect1):
                     outside_people = True
                 if outside_people is False:
                   encapsulated = False
                   break

              if encapsulated is False:
                tstart = time.time()
                bboxes = face_tracker.update(data['gray'])[1]
                print("\tface tracker",time.time() - tstart)
                if time.time() - tstart > 0.1:
                  print('warning face_tracker_done',time.time() - tstart)

                good_rects = 0
                i = 0
                for bbox in bboxes:
                  face_rects[i] = scaleRect(bbox,1/mser_scaler,1/mser_scaler,face_rects[i][4])
                  if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                    good_rects += 1
                  i += 1
                if good_rects == 0:
                  face_tracker = None

          #print("output_6",time.time() - start)


#
# TODO: Figure out some scheme for non max supression 
#  https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
#
          if len(logo_rects) > 0:
            for rect in logo_rects:
              label = "%s: %d" % (rect[2],int(rect[1]*100))
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(image_np, (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (255,0,0), cv2.FILLED)
              if frame < rect[3] + FPS/2:
                cv2.rectangle(image_np, (rect[0][0], rect[0][1]), (rect[0][2],rect[0][3]), (255, 0, 0), 1)
              cv2.putText(image_np, label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

          if len(ssd_rects) > 0:
            #setLabel(image_np, "OBJECT1: %d" % len(ssd_rects),295)
            for rect in ssd_rects:
              label = rect[2].title()
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]

              written = False
              rect_center = rectCenter(rect[0])
              for orect in oi_rects:
                if orect[0][0] < rect_center[0] and orect[0][2] > rect_center[0] and orect[0][1] < rect_center[1] and orect[0][3] > rect_center[1]:
                  # if two things line up, I should describe the color (car Car, bird Bird)
                  #orect[2] = rect[2] +' '+ orect[2].rsplit(None, 1)[-1]
                  written = True
                  break
              for krect in kitti_rects:
                if written == False and krect[0][0] < rect_center[0] and krect[0][2] > rect_center[0] and krect[0][1] < rect_center[1] and krect[0][3] > rect_center[1]:
                  # if two things line up, I should describe the color (car Car, bird Bird)
                  #krect[2] = rect[2].title() +' '+ krect[2].rsplit(None, 1)[-1]
                  written = True

              for srect in ssd_rects:
                if rect == srect:
                  continue

                if srect[0][0] < rect_center[0] and srect[0][2] > rect_center[0] and srect[0][1] < rect_center[1] and srect[0][3] > rect_center[1]:
                  # if two things line up, I should describe the color (car Car, bird Bird)
                  written = True
                  break

              if written == False and rect[1] > 0.3:
                cv2.rectangle(image_np, (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (255,0,255), cv2.FILLED)
                # only display rects on things that deserve it
                #if frame < rect[3] + FPS/2 and rect[1] > 0.3:
                #  cv2.rectangle(image_np, (rect[0][0], rect[0][1]), (rect[0][2],rect[0][3]), (255, 0, 255), 1)
                cv2.putText(image_np, label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

          if len(kitti_rects) > 0:
            #setLabel(image_np, "OBJECT3: %d" % len(kitti_rects),355)
            for rect in kitti_rects:
              label = rect[2].title()
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(image_np, (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (255,0,0), cv2.FILLED)
              cv2.putText(image_np, label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


          if len(oi_rects) > 0:
            #setLabel(image_np, "OBJECT2: %d" % len(oi_rects),315)
            for rect in oi_rects:
              label = rect[2].title()
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(image_np, (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (0,255,255), cv2.FILLED)
              #if frame < rect[3] + FPS/2:
              #  cv2.rectangle(image_np, (rect[0][0], rect[0][1]), (rect[0][2],rect[0][3]), (0, 255, 255), 1)
              cv2.putText(image_np, label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

          if len(person_rects) > 0:
            #setLabel(image_np, "PEOPLE: %d" % len(person_rects),335)
            appearances = defaultdict(int)
            for rect in person_rects:
              if len(person_rects) > 3 and rect[1] < 0.6:
                break
              if len(person_rects) > 2 and rect[1] < 0.5:
                break
              if len(person_rects) > 1 and rect[1] < 0.4:
                break
              if len(person_rects) > 0 and rect[1] < 0.3:
                break
              label = "%s" % rect[2]
              appearances[label] += 1
              # Don't write 2 people with the same label
              if label[:4] == 'CELE' and appearances[label] > 1:
                continue
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(image_np, (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (0,255,0), cv2.FILLED)
              #if frame < rect[3] + FPS/2 and len(person_rects) < 3:
              #  cv2.rectangle(image_np, (rect[0][0], rect[0][1]), (rect[0][2],rect[0][3]), (0, 255, 0), 1)
              cv2.putText(image_np, label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

          if len(face_rects) > 0:
            #setLabel(image_np, "FACE: %s" % base64.b64encode(rect[4]).decode('utf-8'),275)
            i = 0
            for rect in face_rects:
              label = str(rect[4])
              written = False

              # overwrite the person label with the ID 
              rect_center = rectCenter(rect)
              for prect in person_rects:
                if prect[0][0] < rect_center[0] and prect[0][2] > rect_center[0] and prect[0][1] < rect_center[1] and prect[0][3] > rect_center[1]:
                  prect[2] = label
                  written = True
                  break

              if written is False:
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                if frame < rect[3] + FPS/2 and face_hulls[i] is not None:
                  cv2.polylines(image_np, face_hulls[i], False, (0, 255, 0), 1)
                cv2.rectangle(image_np, (rect[0], rect[3]+text_size[1]+10), (rect[0]+text_size[0]+10, rect[3]), (0,255,0), cv2.FILLED)
                cv2.putText(image_np,label,(rect[0]+5,rect[3]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
              i += 1

          if len(face_hulls) > 0:
            for hull in face_hulls:
              cv2.polylines(image_np, [hull], False, (0, 255, 0), 1)

# TODO: Use this to inform the commercial vs programming
          if len(text_rects) > 0:
            for rect in text_rects:
              if rect[2] >= 0:
                label = '%s : %d' % (rect[1],rect[2])
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                cv2.rectangle(image_np, (rect[0][0],rect[0][1]),(rect[0][0]+text_size[0]+10,rect[0][1]+text_size[1]+10), (255,255,255), cv2.FILLED)
                cv2.putText(image_np, label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            shot_textprint += '1'
          else:
            shot_textprint += '0'

          if len(text_hulls) > 0:
            for hull in text_hulls:
              cv2.polylines(image_np, [hull], False, (255, 255, 255), 1)

          #if len(gray_cnts) > 0:
          #  cv2.polylines(image_np,[p.reshape(-1, 1, 2) * int(1/mser_scaler) for p in gray_cnts], False, (0, 255, 0), 1)
          if len(motion_hulls) > 0:
            #for hull in motion_hulls:
            #  cv2.polylines(image_np, [hull], False, (0, 255, 0), 1)
            shot_videoprint += '1'
          else:
            shot_videoprint += '0'

          if logo_detect:
            setLabel(image_np, "LOGO: %s %s" % (logo_detect,logo_detect_count),20)
          #if scores[0][0] > 0.2:
          #  setLabel(image_np, "DEEP: %s %.3f%s" % (category_index[classes[0][0]]['name'],scores[0][0],'%'),215)
          #setLabel(image_np, "COMM: %s" % com_detect,50)


          if scene_detect > frame - FPS:
              setLabel(image_np, "SCENE DETECT: %s" % frame_type,90)
          elif shot_detect > frame - FPS: 
              setLabel(image_np, "SHOT DETECT: %s" % frame_type,90)
          elif subtle_detect > frame - FPS: 
            setLabel(image_np, "SUBTLE DETECT",90)
          elif motion_detect > frame - FPS: 
            setLabel(image_np, "MOTION DETECT",90)

          #if data.get('audio') and data['silence'] > 0:
          #  print('writing test file')
          #  fd = open('/tmp/test.wav','wb')
          #  fd.write(data['audio'])
          #  fd.close()

          if data.get('audio'):
            # TODO:  check the audio level based on a few thing
            # this seems to be the halfway mark on my computer
            if data['audio_level'] >= 500:
              shot_audioprint += '1'
            else:
              shot_audioprint += '0'

            # get peaks and valleys, create hash based on them (wavelets)

# voiceprint
          if data['speech_level'] > 0.6:
            #setLabel(image_np, "SPEAK: %s" % data['speech_level'],255)
            #setLabel(image_np, "SPEAK: %s" % data['speech_level'],255)
            shot_voiceprint += '1'
          else:
            shot_voiceprint += '0'

          if shot_summary:
            setLabel(image_np, "DESC: %s" % shot_summary.title(),435)
          #if frame_caption:
          #  setLabel(image_np, "CAPT: %s" % frame_caption,455)
          if frame_transcribe:
            setLabel(image_np, "TSCB: %s" % frame_transcribe,475)

          #setLabel(image_np, "VIDEO: %s" % phash_bits(shot_videoprint),375)
          #setLabel(image_np, "TEXTP: %s" % phash_bits(shot_textprint),395)
          #setLabel(image_np, "VOICE: %s" % phash_bits(shot_voiceprint) ,435)
          #setLabel(image_np, "AUDIO: %s" % phash_bits(shot_audioprint),415)

          if frame % 30 == 0:
            cap_buf = phash_bits(scene_shotprint)
          setLabel(image_np,"Title: Unnamed %s (%.1fs)" % (prev_scene['type'],float((frame - last_scene + break_count) / FPS)),550)
          setLabel(image_np,' Sig: %s' % cap_buf  ,575)
          if scene_caption:
            setLabel(image_np, ' ...%s' % scene_caption[-80:],610)
          if prev_scene is not None and prev_scene['length'] > 0:
            setLabel(image_np, "Previous: Unnamed %s (%.1fs)" % (prev_scene['type'],prev_scene['length']),640)
            setLabel(image_np, " Sig: %s" % (prev_scene['fingerprint']),665)
            if prev_scene['caption']:
              setLabel(image_np, ' ...%s' % prev_scene['caption'][-80:],690)

          #if time.time() - start > 0.1:
          #  print("output_done",time.time() - start)

# 
#  Letterbox and pillarbox the image before displaying it
#   Probably not everyone wants this
#
          tstart = time.time()
          cast_frame = np.zeros((HEIGHT,WIDTH,3),np.uint8)
          x = int((WIDTH - data['width'])/2)
          y = int((HEIGHT - data['height'])/2) 
          cast_frame[y:data['height']+y,x:data['width']+x:] = image_np
          if cvs is not None and cvs.device is not None:
            cvs.cast(cast_frame)
            if time.time() - tstart > 0.1:
              print("warn slow writing",time.time() - tstart)

# 
# Pause a little longer between frames if we are going faster than
#  the FPS as defined
#
          if frame > 100 and local_fps > FPS:
            tmp = int(local_fps-FPS) * 10 
            if tmp > 30:
              tmp = 30
            waitkey += tmp

          #if fvs.Q.qsize() < 2000 or fvs.Q.qsize() > 900:
          if True:
            setLabel(cast_frame, "FBUF: %d (%dms)" % (fvs.Q.qsize(),waitkey),115)
            setLabel(cast_frame, "FRAM: %d (%.3ffps)" % (fps_frame,fps_frame / (time.time() - start_time)),135)
            setLabel(cast_frame, "REAL: %s" % filetime,155)
            setLabel(cast_frame, "VIEW: %s" % frametime,175)

          cv2.imshow('Object Detection',cast_frame)
          if fvs.image:
             # an hour?
             waitkey = 60*60*1000
          key = cv2.waitKey(waitkey) 
          if 'c' == chr(key & 255):
            fvs.stop()
            #fvs.Q = None
            #fvs.t.terminate()
            #fvs.ocr.t.terminate()
            while fvs.Q.qsize() > 0:
               fvs.Q.get()
            while fvs.ocr.input.qsize() > 0:
               fvs.ocr.input.get()
            while fvs.ocr.output.qsize() > 0:
               fvs.ocr.output.get()
            fvs.t.join()
            print('done')
            break
          elif 's' == chr(key & 255):
            fvs.ff = True
            continue
          elif ' ' == chr(key & 255):
              cv2.destroyWindow('Object Detection')
              r = cv2.selectROI('Select Station Logo',original_frame,False,False)
              print("r:",r)
              if r[0] > 0 and r[3] > 0:
                folder = datetime.datetime.utcnow().strftime('%Y%m%d')
                filename = "%s.jpg" % datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
                # Need to pull chan out of URL, or maybe from recent logos detected
                #directory = os.path.join("/home/kristopher/tf_files/images/tv/",chan,folder)
                #if not os.path.exists(directory):
                #  os.makedirs(directory)
                #cv2.imwrite(os.path.join(directory,filename),original_frame)
            # take this and put it into the database

              cv2.destroyWindow('Select Station Logo')
              cv2.imshow('Object Detection', image_np)

          elif 'q' == chr(key & 255):
            fvs.stop()
            sys.exit()

         # TODO: put some alarms around this, so if a source times out we know it
         #loop through
          if fvs.stream[:4] == 'rtsp' and frame > 1500:
            fvs.stop()
            break
          elif frame > 10000:
            fvs.stop()
            break


