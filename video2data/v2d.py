#!/usr/bin/python3
# coding: utf-8
#
#

FPS = 30
WIDTH = 1920
HEIGHT = 1080
#WIDTH = 1280
#HEIGHT = 720
WIDTH = 800
HEIGHT = 450
SAMPLE = 44100
SCALER = 0.2

import os, sys, math
import tensorflow as tf

from utils2 import image as v2dimage
from utils2 import sig as v2dsig
from utils2 import realtime as realtime
from utils2 import log as log
from utils2 import var as v2dvar

import streams.read
# Load up the sources 
sources = [line.rstrip('\n') for line in open('sources.txt')]

#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("source", help="path to your media file")
#args = parser.parse_args()
#if not sources and args.source:
#  sources = [args.source]


# 3.4.0
import cv2
import numpy as np
import wave

image_np = cv2.imread("data/test.png")
image_np_expanded = np.expand_dims(image_np, axis=0)

import models.face
faces = None
#faces = models.face.nnFace().load()
#faces.test(image_np)

#import models.im2txt
#im2txt = None
#im2txt = models.im2txt.nnIm2txt().load()
#im2txt.test(image_np)

#import models.text
text = None
#text = models.text.nnText().load()
#text.test(image_np)

import models.ssd
#ssd = None
ssd = models.ssd.nnSSD().load()
ssd.test(image_np_expanded)

import models.vehicle
vehicle = None
#vehicle = models.vehicle.nnVehicle().load()
#vehicle.test(image_np_expanded)

import models.objects
# disabled because slow
objects = None
#objects = models.objects.nnObject().load()
#objects.test(image_np_expanded)

import models.contrib.deepad
logo = None
#logos = models.contrib.deepad.nnLogo().load()
#logos.test(image_np_expanded)


# and playing with fifo buffers
import os, sys, math

import datetime

# A few subprocesses open pipes into the 
import subprocess
import time
import re
# encode 128D vectors for facial fingerprints
import base64

# I want to set a dict with a key value without initializing the key
from collections import defaultdict

# this is a pretty good audio playing package.  unblink queues up a 
#  bunch of audio as it comes in from the frames.  The audio needs to be joined
# and preferably on breaks. 
import pygame
pygame.init()
#  44100 samples, 16 bit, mono.  I considered stereo but the speech to text 
# translators all want mono anyway.  We can always add additional channels
pygame.mixer.init(SAMPLE, -16, 1)

# initialize the tracker
people_tracker = cv2.MultiTracker_create()
oi_tracker = cv2.MultiTracker_create()
face_tracker = cv2.MultiTracker_create()
kitti_tracker = cv2.MultiTracker_create()

#import streams.cast
cvs = None
#cvs = streams.cast.CastVideoStream().load(WIDTH,HEIGHT)


all_stuff = []
if ssd is not None:
  for item in ssd.ccategory_index:
    all_stuff.append(ssd.ccategory_index[item]['name'])

bstart = time.time()

def setLabel(im, label, y):
  fontface = cv2.FONT_HERSHEY_SIMPLEX
  scale = 0.6
  thickness = 1

  text_size = cv2.getTextSize(label, fontface, scale, thickness)[0]
  cv2.rectangle(im, (10, y - text_size[1] - 5), (10 + text_size[0], y + 5), (0,0,0), cv2.FILLED)
  cv2.putText(im, label ,(10, y),fontface, scale, (0, 255, 0), thickness)

def parse_audfprint(line):
  clean = line.rstrip().decode('ascii', errors='ignore')
  prev_lookup = {}
  for line in clean.splitlines(): 
    mo = re.search("raw hashes as .+?([^\/]+?)-360-.+?.mp4",line)
    if mo and mo.group():
      prev_lookup['name'] = mo.group(1).replace('-',' ').title()
  return prev_lookup
  
if 1==1:
  if 1==1:

    all_faces = []
    while True:
      for source in sources:
        if source[0] in ['#',' ']:
          continue

        fvs = streams.read.FileVideoStream(source).load(WIDTH,HEIGHT,FPS,SAMPLE)
        print('source: %s' % source)
        print('stopped',fvs.stopped)

        last_frame = None
        prev_scene = v2dvar.new_scene(0)
        prev_prev_scene = v2dvar.new_scene(0)
        prev_shot = v2dvar.new_shot(0)

        frame_caption = ''
        caption_end = ''
        last_caption = ''

        scene_count = 0
        scene_detect = 0
        audio_detect = 0
        motion_start = 0
        subtle_detect = 0
        last_transcribe = 0
        frame = 0
        last_shot = 0

        fps_frame = 0
        start_time = time.time()
        channel1 = pygame.mixer.Channel(0)

        tmp_pipes = []
        audfprint_queues = []

        oi_rects = []
        kitti_rects = []
        ssd_rects = []
        face_rects = []
        face_hulls = []
        motion_hulls = []
        motion_cnts = []

        oi_frame = 0
        ssd_frame = 0
        kitti_frame = 0
        face_frame = 0

        last_play_audio = bytes()

        while fvs.stopped.value == 0:
          if fvs.Q.qsize() < 1:
            if fvs.t and fvs.t.is_alive() is False:
              print('done')
              break
            #print("READ<",time.time())
            time.sleep(0.1)
            continue

# Main Reader from other processing thread
          data = fvs.read()

# play the audio
# TODO: rename cast_audio 
          if fvs.display and data.get('play_audio') is not None:    
            last_play_audio = data['play_audio']
            play_audio = pygame.mixer.Sound(data.get('play_audio'))
            # don't play the first packet. annoying pop
            if data['frame'] > 1:
              sound1 = pygame.mixer.Sound(play_audio)
              channel1.queue(sound1)
            prev_scene['play_audio'] += data['play_audio']
            prev_scene['segment'] += 1
         

          start = time.time()
          waitkey = realtime.smoothWait(fvs)
          fvs.ff = False

          data['frame'] = frame
          data['frametime'] = datetime.datetime.utcnow().utcfromtimestamp(frame / fvs.fps).strftime('%H:%M:%S,%f')[:-3]
          local_fps = fps_frame / (time.time() - start_time)
      
          if data.get('caption'):
            # sometimes captions get duplicated (truncated, appended to the previous caption).  It's a function of the transcriber.  
            #  see if we can filter those out
            tmp_caption = ''
            for i in reversed(range(3,len(data['caption']))):
              if frame_caption[-i:] == data['caption'][0:i]:
                 tmp_caption = data['caption'][i:]
                 break
            if tmp_caption is '':
              tmp_caption = data['caption']
            frame_caption = data['caption']
            print('\t\t[%s] CC: %s' % (data['caption_time'],tmp_caption))


          if last_frame and last_frame['text_hulls']:
            data['text_hulls'] = last_frame['text_hulls']
          
          if data.get('transcribe'):
            last_transcribe = frame

          if last_frame and 'BLANK' not in last_frame['shot_type'] and last_frame['frame'] < last_frame['shot_detect'] - 2:
            prev_shot['last_frame'] = last_frame['small']
            prev_scene['closing'] = prev_shot['last_frame']
            prev_shot['closing'] = prev_shot['last_frame']

          if data.get('shot_detect') == frame and last_frame is not None and prev_shot.get('last_frame') is not None:
            if prev_shot['closing'] is None:
              prev_shot['closing'] = last_frame['small']
            if prev_scene['opening'] is None:
              prev_scene['opening'] = prev_shot['closing']

            prev_shot['end'] = frame-1
            prev_shot['break_type'] = data['break_type']
            prev_shot['shottime'] = datetime.datetime.utcnow().utcfromtimestamp((prev_shot['start']) / data['fps']).strftime('%H:%M:%S,%f')[:-3]
            prev_shot['length'] = float((prev_shot['end'] - prev_shot['start']) / FPS)
            log.shot_recap(prev_shot)

          if frame % 60 == 0:
            start_time = time.time()
            fps_frame = 0

#  SCENE BREAK DETECTOR
# 
          if prev_scene['network_detect'] > 10:
            prev_scene['com_detect'] = 'Programming'
            prev_scene['title'] = 'Regular Programming'
          if prev_scene['length'] > 300:
            prev_scene['com_detect'] = 'Programming'
            prev_scene['title'] = 'Regular Programming'

          prev_scene['end'] = frame-1
          prev_scene['scenetime'] = datetime.datetime.utcnow().utcfromtimestamp((prev_scene['start']) / FPS).strftime('%H:%M:%S,%f')[:-3]
          if data.get('scene_detect') == frame and last_frame is not None:
            audfprint_queues = []
            data['scene_detect'] = frame
            if prev_scene['closing'] is None:
              prev_scene['closing'] = last_frame['small']
            prev_scene['sbd'] = data['sbd']
            prev_scene['length'] = float((prev_scene['end'] - prev_scene['start']) / FPS)
            prev_scene['break_type'] = prev_shot['break_type']
            prev_scene['last_frame'] = prev_shot.get('last_frame')
            prev_scene['closing'] = prev_shot['closing']

# TODO: move to DA contrib
            if prev_scene['length'] < 2:
              prev_scene['com_detect'] = 'Blank'
              prev_scene['title'] = 'Dead Air'

            cr = 100*prev_scene['shotprint'].count('1') / len(prev_scene['shotprint'])
            if int(round(prev_scene['length'])) in [5,10,15] and prev_scene['com_detect'] is 'Programming':
              prev_scene['com_detect'] = 'Network Promo'
              prev_scene['title'] = 'Network Promo'
            if cr >= 0.9 and int(round(prev_scene['length'])) in [10,15,30,45,60]:
              prev_scene['com_detect'] = 'Commercial'
            elif prev_prev_scene['com_detect'] == 'Commercial' and int(round(prev_scene['length'])) in [15,30,45,60]:
              prev_scene['com_detect'] = 'Commercial'
            if prev_scene['length'] > 90 and int(round(prev_scene['length'])) % 15 != 0:
              prev_scene['com_detect'] = 'Programming'
         
            if prev_scene['com_detect'] == 'Commercial' and prev_scene['title'] == '':
              prev_scene['title'] = 'Unknown Commercial'
            if prev_scene['com_detect'] == 'Programming' and prev_scene['title'] == '':
              prev_scene['title'] = 'Regular Programming'

            if prev_scene['title'] == '':
              prev_scene['title'] = 'Unknown Segment'

            log.scene_recap(prev_scene)

            if prev_scene['length'] > 2:
              prev_prev_scene = prev_scene

            prev_scene = v2dvar.new_scene(frame)
            prev_shot = v2dvar.new_shot(frame)

            caption_end = ''
            frame_caption = ''
            scene_count += 1


          if prev_prev_scene['opening'] is not None:
            y = data['small'].shape[0]
            x = data['small'].shape[1]
            if x+20 < data['show'].shape[1]:
              data['show'][data['height']-y-20:data['height']-20,20:x+20] = prev_prev_scene['opening'].copy()
          if prev_prev_scene['closing'] is not None:
            y = data['small'].shape[0]
            x = data['small'].shape[1]
            if 2*x+40 < data['show'].shape[1]:
              data['show'][data['height']-y-20:data['height']-20,x+40:2*x+40] = prev_prev_scene['closing'].copy()

          if prev_scene['opening'] is not None:
            y = data['small'].shape[0]
            x = data['small'].shape[1]
            if 3*x+60 < data['show'].shape[1]:
              data['show'][data['height']-y-20:data['height']-20,2*x+60:3*x+60] = prev_scene['opening'].copy()
          if prev_scene['closing'] is not None:
            y = data['small'].shape[0]
            x = data['small'].shape[1]
            if 4*x+80 < data['show'].shape[1]:
              data['show'][data['height']-y-20:data['height']-20,3*x+80:4*x+80] = prev_scene['closing'].copy()

          if data.get('shot_detect') == frame:
            prev_shot = v2dvar.new_shot(frame)

##################
#
# SHOT DETECTOR
#
# if we detect this is a new shot, then all of the detection needs to be redone
          #print("output_0",time.time() - start)
          if data.get('shot_detect') == frame:
            prev_scene['text'].extend(data['text_hulls'])
            for rect in oi_rects:
              if rect[1] > 0.8:
                prev_scene['stuff'].append(rect[2])
            for rect in data['ssd_rects']:
              if rect[1] > 0.8:
                prev_scene['stuff'].append(rect[2])
            for rect in kitti_rects:
              if rect[1] > 0.7:
                prev_scene['vehicles'].append(rect[2])
            for rect in kitti_rects:
              if rect[1] > 0.8:
                prev_scene['vehicles'].append(rect[2])
            # reset all the OCR if we change shot
            for pipe in tmp_pipes:
              pipe[0].kill()
            tmp_pipes = []
            face_rects = []
            face_hulls = []
            ssd_rects = []
            oi_rects = []
            kitti_rects = []
            motion_cnts = []
            motion_hull = []
            data['text_hulls'] = []
            data['text_rects'] = []
            oi_tracker = None
            kitti_tracker = None
            face_tracker = None

            prev_scene['shotprint'] += '1'
            #print("shot detect!",frame,shot_detect,motion_detect,subtle_detect)
          else:
            prev_scene['shotprint'] += '0'


################
#
# Audio fingerprint.  This is where we take a segment of a clip, and map the 
#  fingerprint.  Check the fingerprint against databases (dejavu) to see if 
#  this clip exists already.  Can be pretty low quality fingerprint, as long as it is 
#  fast.  Also, it's OK to wait a few seconds.  Longer segments take less storage to fingerprint
# 
###############
#
#  Text is the highest indicator of advertising. Detects individual letters
#  Runs a neural network underneath
# 
          if text is not None and ((frame == data.get('shot_detect')+1 or frame == data.get('shot_detect')+7 or frame == data.get('shot_detect') + 37) or fvs.image):
            old_text_rects = []
            if last_frame and last_frame.get('text_rects'):
              old_text_rects = last_frame.get('text_rects')
            start = time.time()
            if text is not None:
              data['text_hulls'] = text.test(data['rframe'])
              if len(data['text_hulls']) > 0:
                for box in data['text_hulls']:
                   (x,y,w,h) = cv2.boundingRect(box)
                   if w < data['width'] * 0.01:
                     print('skipping small text',w,data['width'])
                     continue
                   if h < data['height'] * 0.01:
                     print('skipping small text',h,data['height'])
                     continue

                   detected_label = '?'
                   detected_frame = frame
                   detected_confidence = 0
                   rect_center = v2dimage.cntCenter(box)
                   # if its close to horizontal, don't rotate it.  just extract with some extra
                   # consider adding extras 
                   for trect in old_text_rects:
                     if trect[0][0] < rect_center[0] < trect[0][2] and trect[0][1] < rect_center[1] < trect[0][3]:
                       detected_confidence = trect[2]
                       detected_label = str(trect[1])
                       detected_frame = trect[3]
                       # consider reading from subprocess here
                       break
                   # don't OCR every frame (unless this is an image)
                   print('frame',frame,detected_frame,detected_confidence)
                   if fvs.image or (detected_confidence == 0 and len(tmp_pipes) < 16):
                     tmp = v2dimage.hullHorizontalImage(data['rframe'],box[0])
                     tstart = time.time()
                     # do this as a subprocess, and then rejoin on the read later.  
                     #   libjpeg 8d (libjpeg-turbo 1.5.2) : libpng 1.6.34 : libtiff 4.0.8 : zlib 1.2.11
                     _,img = cv2.imencode(".bmp",tmp)
                     # TODO: don't process images I already did...
                     sp = subprocess.Popen(['tesseract','stdin','stdout','--oem','1','--psm','13','-l','eng','config/tess.config'], stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                     sp.stdin.write(img)
                     sp.stdin.flush()
                     sp.stdin.close()
                     tmp_pipes.append([sp,[[x,y,w+x,y+h],detected_label,detected_confidence,detected_frame]])
                   else:
                     data['text_rects'].append([[x,y,w+x,y+h],detected_label,detected_confidence,detected_frame])


##########
# High FPS object detector. Look for stuff that will inform other neural networks
#  Do you see people?  OK, then we can run Dlib fingerprint
#
# just do this once per shot, or every 30 seconds or so
#  doesn't have to be all the time
# mobilenet, then dlib if we see people
#
          #if ssd is not None and ssd.sess is not None and (data.get('shot_detect') and (frame == data.get('shot_detect') + 3 or fvs.image or (frame <= motion_start + 4 and frame % 5 == 0)) or frame % 31 == 0):
          #if ssd is not None and ssd.sess is not None and (frame <= motion_start + 4 and frame % 5 == 0)) or frame % 31 == 0):
          if ssd is not None and ssd.sess is not None and (data.get('shot_detect') and (frame == data.get('shot_detect') + 3 or fvs.image or (frame <= motion_start + 4 and frame % 5 == 0)) or frame % 7 == 0):
            ssd_frame = frame

            #  It doesn't really make sense to run this
            #  the SSD module is generally faster than the motion tracker
            people_tracker = cv2.MultiTracker_create()
            kitti_tracker = cv2.MultiTracker_create()
            data = ssd.run(data,last_frame)
            #for human_rect in data['human_rects']:
              #print("data:",human_rect);
              #ok = people_tracker.add(cv2.TrackerKCF_create(),frame['small'],v2dimage.scaleRect(v2dimage.shiftRect(human_rect[0][i]),frame['width']*SCALER,frame['height']*SCALER))
          elif frame != data['shot_detect'] and last_frame:
            data['human_rects'] = last_frame['human_rects']
            data['ssd_rects'] = last_frame['ssd_rects']
            data['vehicle_rects'] = last_frame['vehicle_rects']

            
########
# High FPS Logo detector. This is one of our proprietary detectors.  Logos help us spot commercials
#  workers, endorsements, and lots of other things.  Ours is trained on 16 classes including 
#  dental care, alcohol, and network brands.
#
          #if (frame == data.get('shot_detect')+3 or (frame >= motion_start + 10 and frame < motion_start + 10 and frame % 10 == 0) or (frame > subtle_detect and frame < subtle_detect + 30 and frame % 10 == 0) or frame % 601 == 0):
          #  # TODO: move the timer inside the model
          #  data = logos.run(data,last_frame)
          #  if data['com_detect'] == 'Programming':
          #    prev_scene['network_detect'] += 1


###############
#
# FasterRCNN Slow detector. (0.8s) Using Open Images, tell if what is on the scene is one of 546 items. 
#
#   Only run on "deep" scenes.  Where the shot hasn't changed in a long time
# 
#  WARNING: This takes TWELVE SECONDS (12) to run the first time on my computer.  
# 
          if False and objects.sess is not None and (fvs.image or frame == data.get('shot_detect') + 3):
            oi_frame = frame
            tstart = time.time()
            # ive seen this take 12s the first time it loads, then its fast
            #  usually 0.8s otherwise
            image_np_expanded = np.expand_dims(data['rframe'], axis=0)
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
                #human_rects.append([v2dimage.scaleRect(v2dimage.shiftRect(oboxes[0][i]),data['width'],data['height']),oscore,'Person',frame])
                # hopefully people_tracker exists?
                #if people_tracker:
                #  ok = people_tracker.add(cv2.TrackerKCF_create(),data['gray'],v2dimage.scaleRect(v2dimage.shiftRect(oboxes[0][i]),data['width']*SCALER,data['height']*SCALER))
              elif ocategory_index[oclasses[0][i]]['name'] in ['Car','Bus','Truck','Boat','Airplane','Vehicle','Tank','Vehicle','Van','Land vehicle','Limousine','Watercraft','Barge']:
                if oscore < 0.2:
                  continue
                # TODO: this only persists as long as the ssd tracker doens't overwrite it. It does help with the weighting on the next ssd pass
                kitti_rects.append([v2dimage.scaleRect(v2dimage.shiftRect(oboxes[0][i]),data['width'],data['height']),oscore,ocategory_index[oclasses[0][i]]['name'].lower(),frame])
                ok = kitti_tracker.add(cv2.TrackerKCF_create(),data['gray'],v2dimage.scaleRect(v2dimage.shiftRect(oboxes[0][i]),data['width']*SCALER,data['height']*SCALER))
              elif len(oi_rects) < 5:
              # TODO: Check if this is already in a rect -- if so do not append it
                oi_rects.append([v2dimage.scaleRect(v2dimage.shiftRect(oboxes[0][i]),data['width'],data['height']),oscore,ocategory_index[oclasses[0][i]]['name'].lower(),frame])
                ok = oi_tracker.add(cv2.TrackerKCF_create(),data['small'],v2dimage.scaleRect(v2dimage.shiftRect(oboxes[0][i]),data['width']*SCALER,data['height']*SCALER))
              i += 1
            #if time.time() - tstart > 1:
            #  print("warning oi_done",time.time() - start)

###################33
#   Scene Describer
#   Takes the image and turns it into text.  This can be used to trigger other detectors or vice versa
#    Takes 0.4s 
          #if len(oi_rects) > 0 and (frame == shot_detect+37 or fvs.image or (frame > shot_detect + 173 and frame % 173 == 0)):
          #if len(oi_rects) > 0 and (frame == shot_detect+4 or fvs.image):
          # consider launching this as a process, and then joining on it later 
          if (data.get('shot_detect') and frame == data.get('shot_detect')+4) or fvs.image:
            tstart = time.time()

            include_labels = []
            exclude_labels = ['wii']
            # keep gender neutral and low
            exclude_labels.extend(['men','women','boys','girls','child','guy','girl','boy','man','woman','children','male','female','lady','kid','kids','skateboarder','gentleman'])
            if len(data['human_rects']) == 0 or data['human_rects'][0][1] < 0.5:
              exclude_labels.extend(['person','people'])
            elif len(data['human_rects']) == 1 and data['human_rects'][0][1] > 0.6:
              include_labels.extend(['person'])
            elif len(data['human_rects']) > 1 and data['human_rects'][0][1] > 0.5 and data['human_rects'][1][1] > 0.5:
              include_labels.extend(['people'])
            else:
              exclude_labels.extend(['person','people'])

            if len(kitti_rects) > 0:
              for i, rect in enumerate(kitti_rects):
                include_labels.extend(rect[2].split())

            if len(oi_rects) > 0:
              for i, rect in enumerate(oi_rects):
                include_labels.extend(rect[2].split())
            #    print('oi_rect',i,rect[1],rect[2])
            else:
              for i, rect in enumerate(data['ssd_rects']):
                include_labels.extend(rect[2].split())

            if 'cell phone' not in include_labels:
              exclude_labels.extend(['mobile','cellphone','cell'])


            appearances = defaultdict(int)
            for curr in include_labels:
              appearances[curr] += 1
            for curr in include_labels:
              # naive nlp right now
              if appearances[curr] > 1 and curr[:-1] is not 's':
                include_labels.extend(['%ss' % curr])

            exclude_labels.extend(list(set(all_stuff) - set(include_labels)))
         
            #if im2txt:
            #  prev_shot['summary'] = im2txt.run(data['rframe'].copy(),include_labels,exclude_labels)
            #  prev_scene['summary'] += prev_shot['summary'] + ' '


###########
# Motion tracking.  Do this instead of detecting an object
#  TODO: Break this up 
#
          if waitkey > 8:
            # we only need to motion detect if an object we detect is inside a motion contour. It is easy to do that, so we might as well
            # TODO: Right now I am only tracking inside a person.  I should track near
            # note: don't run if there are a lot of people, it gets computationally intense
            if people_tracker is not None and data.get('human_rects') is not None and len(data['human_rects']) > 0 and frame > ssd_frame and frame % len(data['human_rects']) == 0 and len(data['human_rects']) < 5:
              encapsulated = False
              for cnt in motion_cnts: 
                 rect_center = v2dimage.cntCenter(cnt)
                 for person_rect in data['human_rects']:
                   w = person_rect[0][2] - person_rect[0][0]
                   h = person_rect[0][3] - person_rect[0][1]
                   # TODO: sometimes it tracks too much.  Slow it down if the person is just idling in place
                   if w*h/(data['width']*data['height']) < 0.3 and v2dimage.pointInRect(rect_center,v2dimage.scaleRect(person_rect[0],SCALER,SCALER)) and len(data['human_rects']) < 2:
                     encapsulated = True
                     break
                   if w*h/(data['width']*data['height']) < 0.4 and v2dimage.pointInRect(rect_center,v2dimage.scaleRect(person_rect[0],SCALER,SCALER)) and len(data['human_rects']) < 3:
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
                  data['human_rects'][i][0] = v2dimage.scaleRect(bbox,1/SCALER,1/SCALER)
                  if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                    good_rects += 1
                  i += 1
                if good_rects == 0:
                  people_tracker = None
                  data['human_rects'] = []

            # this makes the most sense to track
            #  put cars in separate group, as they are high speed
            if kitti_rects is not None and len(kitti_rects) > 0 and frame > ssd_frame:
              encapsulated = False
              for cnt in motion_cnts: 
                 rect_center = v2dimage.cntCenter(cnt)
                 for car_rect in kitti_rects:
                   w = car_rect[0][2] - car_rect[0][0]
                   h = car_rect[0][3] - car_rect[0][1]
                   if w*h/(data['width']*data['height']) < 0.3 and v2dimage.pointInRect(rect_center,v2dimage.scaleRect(car_rect[0],SCALER,SCALER)):
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
                  kitti_rects[i][0] = v2dimage.scaleRect(bbox,1/SCALER,1/SCALER)
                  if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                    good_rects += 1
                  i += 1
                if good_rects == 0:
                  kitti_tracker = None
                  kitti_rects = []

            if oi_frame != frame and oi_tracker is not None and oi_rects is not None and len(oi_rects) > 0 and frame % len(oi_rects) == 0:
              encapsulated = False
              for cnt in motion_cnts: 
                 rect_center = v2dimage.cntCenter(cnt)
                 for car_rect in kitti_rects:
                   if v2dimage.pointInRect(rect_center,car_rect[0]):
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
                  oi_rects[i][0] = v2dimage.scaleRect(bbox,1/SCALER,1/SCALER)
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
                   if v2dimage.rectInRect(rect2,rect1):
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
                  face_rects[i] = v2dimage.scaleRect(bbox,1/SCALER,1/SCALER,face_rects[i][4])
                  if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                    good_rects += 1
                  i += 1
                if good_rects == 0:
                  face_tracker = None

          if data.get('contrib_rects') and len(data['contrib_rects']) > 0:
            for rect in data['contrib_rects']:
              label = "%s: %d" % (rect[2],int(rect[1]*100))
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(data['show'], (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (255,0,0), cv2.FILLED)
              cv2.putText(data['show'], label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
              prev_scene['contrib_rects'].append([rect[2],int(rect[1]*100)])

          if len(data['ssd_rects']) > 0:
            for rect in data['ssd_rects']:
              if rect[1] < 0.6 and frame < rect[3] + 10:
                break
              if rect[1] < 0.5 and frame < rect[3] + 20:
                break
              if rect[1] < 0.4 and frame < rect[3] + 40:
                break
              if rect[1] < 0.3 and frame < rect[3] + 50:
                break
              label = rect[2].title()
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]

              written = False
              rect_center = v2dimage.rectCenter(rect[0])
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

              for srect in data['ssd_rects']:
                if rect == srect:
                  continue

                if srect[0][0] < rect_center[0] and srect[0][2] > rect_center[0] and srect[0][1] < rect_center[1] and srect[0][3] > rect_center[1]:
                  # if two things line up, I should describe the color (car Car, bird Bird)
                  written = True
                  break

              if written == False and rect[1] > 0.3:
                cv2.rectangle(data['show'], (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (255,0,255), cv2.FILLED)
                # only display rects on things that deserve it
                cv2.putText(data['show'], label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

          if len(kitti_rects) > 0:
            for rect in kitti_rects:
              label = rect[2].title()
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(data['show'], (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (255,0,0), cv2.FILLED)
              cv2.putText(data['show'], label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


          if len(oi_rects) > 0:
            for rect in oi_rects:
              label = rect[2].title()
              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(data['show'], (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (0,255,255), cv2.FILLED)
              cv2.putText(data['show'], label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

          if len(data['human_rects']) > 0:
            appearances = defaultdict(int)
            for rect in data['human_rects']:
              if rect[1] < 0.3:
                 break
              #if rect[1] < 0.6 and frame < rect[3] + 10:
              #  break
              #if rect[1] < 0.5 and frame < rect[3] + 20:
              #  break
              #if rect[1] < 0.4 and frame < rect[3] + 40:
              #  break

              #if len(data['human_rects']) > 3 and rect[1] < 0.6:
              #  break
              #if len(data['human_rects']) > 2 and rect[1] < 0.5:
              #  break
              #if len(data['human_rects']) > 1 and rect[1] < 0.4:
              #  break
              #if len(data['human_rects']) > 0 and rect[1] < 0.3:
              #  break
              age = frame-rect[3]
              label = "%s %.2f %d" % (rect[2],rect[1],age)
              #label = rect[2]
              appearances[label] += 1
              # Don't write 2 people with the same label
              if label[:4] == 'CELE' and appearances[label] > 1:
                continue
             
              # frame age
              if age > 128:
                age = 128

              text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
              cv2.rectangle(data['show'], (rect[0][0], rect[0][1]), (rect[0][2], rect[0][3]), (0,255-age,0), 2)
              cv2.rectangle(data['show'], (rect[0][0], rect[0][1]+text_size[1]+10), (rect[0][0]+text_size[0]+10, rect[0][1]), (0,255-age,0), cv2.FILLED)
              cv2.putText(data['show'], label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

          if len(face_rects) > 0:
            i = 0
            for rect in face_rects:
              label = str(rect[4])
              written = False

              # overwrite the person label with the ID 
              rect_center = v2dimage.rectCenter(rect)
              for prect in data['human_rects']:
                if prect[0][0] < rect_center[0] and prect[0][2] > rect_center[0] and prect[0][1] < rect_center[1] and prect[0][3] > rect_center[1]:
                  prect[2] = label
                  written = True
                  break

              if written is False:
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                cv2.rectangle(data['show'], (rect[0], rect[3]+text_size[1]+10), (rect[0]+text_size[0]+10, rect[3]), (0,255,0), cv2.FILLED)
                cv2.putText(data['show'],label,(rect[0]+5,rect[3]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
              i += 1


##########
#  subpocess harvestor 
# 
          busy_pipes = []
          for pipe in tmp_pipes:
            # 1ms per poll() ...
            # TODO: shouldnt this be "is None" ???
            if pipe[0].poll() is not None:
              busy_pipes.append(pipe)
              continue
 
            # see if this is ready for reading first.  If so, then read it
            # otherwise skip it and come back later
            tmp_data = pipe[0].stdout.read()
            detected_label = pipe[1][1]
            detected_confidence = pipe[1][2]
            detected_frame = pipe[1][3]
            for line in tmp_data.decode('utf-8').split('\n'):
              col = line.split('\t')
              if len(col) > 10 and col[10].isdigit() and int(col[10]) > 0 and int(col[10]) > detected_confidence:
                detected_confidence = int(col[10])
                detected_label = col[11]
                if detected_confidence > 80 and len(detected_label) > 4 or detected_confidence > 90 and len(detected_label) > 2:
                  prev_scene['scene_words'].append(detected_label)
                  #print('\t\t\tconf',pipe[1][0][2] - pipe[1][0][0],col[8],col[10],col[11])
                #print('\t\tOCR:',detected_label,detected_confidence)
              data['text_rects'].append([pipe[1][0],detected_label,detected_confidence,detected_frame])
          tmp_pipes = busy_pipes    

          if data.get('text_hulls') and len(data['text_hulls']) > 0:
            for hull in data.get('text_hulls'):
              cv2.polylines(data['show'], [hull], False, (255, 255, 255), 1)

# TODO: Use this to inform the commercial vs programming
          if len(data['text_rects']) > 0:
            for rect in data['text_rects']:
              if rect[2] > 80 and '?' not in rect[1]:
                label = rect[1]
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(data['show'], (rect[0][0],rect[0][1]),(rect[0][0]+text_size[0]+10,rect[0][1]+text_size[1]+10), (0,0,0), cv2.FILLED)
                cv2.putText(data['show'], label ,(rect[0][0]+5, rect[0][1]+text_size[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            prev_shot['textprint'] += '1'
          else:
            prev_shot['textprint'] += '0'

          if len(motion_hulls) > 0:
            prev_shot['motionprint'] += '1'
          else:
            prev_shot['motionprint'] += '0'

          if data and data.get('audio') is not None:
            # TODO:  check the audio level based on a few thing
            # this seems to be the halfway mark on my computer
            if data['audio_level'] >= 500:
              prev_shot['audioprint'] += '1'
            else:
              prev_shot['audioprint'] += '0'

            # get peaks and valleys, create hash based on them (wavelets)
# voiceprint
          if data and data.get('speech_level') > 0.6:
            prev_shot['voiceprint'] += '1'
          else:
            prev_shot['voiceprint'] += '0'

          fps_frame += 1   
          frame += 1

          # never put last_frame in data
          last_frame = data.copy()

          #setLabel(data['show'], "FBUF: %d (%dms)" % (fvs.Q.qsize(),waitkey),115)
          #setLabel(data['show'], "FRAM: %d (%.3ffps)" % (data['frame'],fps_frame / (time.time() - start_time)),135)
          #setLabel(data['show'], "VIEW: %s" % data['frametime'],175)
          #setLabel(data['show'], "SBD: %.2f" % data['sbd'],195)
          if data['shot_detect'] > data['frame']:
            setLabel(data['show'], "NEXT: %d" % (int(data['shot_detect']) - int(data['frame'])),215)
          if data['scene_detect'] > data['frame']:
            setLabel(data['show'], "SCENE: %d" % (int(data['shot_detect']) - int(data['frame'])),235)
          if prev_scene['title'] is not '':
            setLabel(data['show'], "CURR: %s" % (prev_scene['title']),255)
          if prev_prev_scene['title'] is not '':
            setLabel(data['show'], "LAST: %s" % (prev_prev_scene['title']),285)
# 
# 
# 
# 
#  Letterbox and pillarbox the image before displaying it
#   Probably not everyone wants this
#
          tstart = time.time()
          cast_frame = np.zeros((HEIGHT,WIDTH,3),np.uint8)
          x = int((WIDTH - data['width'])/2)
          y = int((HEIGHT - data['height'])/2) 
          cast_frame[y:data['height']+y,x:data['width']+x:] = data['show'].copy()
          if cvs is not None and cvs.device is not None:
            cvs.cast(cast_frame)
            if time.time() - tstart > 0.1:
              print("warn slow writing",time.time() - tstart)
# 
# TODO
# Pause a little longer between frames if we are going faster than
#  the FPS as defined. potentially go the other way too
#
          if fvs.display and frame > 100 and local_fps > FPS:
            tmp = (local_fps-FPS)  / 60
            if tmp > 30:
              tmp = 30
            waitkey += int(tmp)

          if fvs.display is False:
            continue

          cv2.imshow('Object Detection',cast_frame)
          if fvs.image:
             # an hour?
             waitkey = 60*60*1000

# WAITKEY
          waitkey = 1

          key = cv2.waitKey(waitkey) 
          if 'c' == chr(key & 255):
            fvs.stop()
            break
          elif 's' == chr(key & 255):
            fvs.ff = True
            continue
          elif ' ' == chr(key & 255):
              cv2.destroyWindow('Object Detection')
              #r = cv2.selectROI('Select Station Logo',data['original'],False,False)
              r = cv2.selectROI('Select Station Logo',cast_frame,False,False)
              if r[0] > 0 and r[3] > 0:
                folder = datetime.datetime.utcnow().strftime('%Y%m%d')
                filename = "%s.jpg" % datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
              cv2.destroyWindow('Select Station Logo')
              cv2.imshow('Object Detection', cast_frame)

          elif 'q' == chr(key & 255):
            fvs.stop()
            sys.exit()

         # TODO: put some alarms around this, so if a source times out we know it
         #loop through
          if fvs.stream[:4] == 'rtsp' and frame > 1500:
            fvs.stop()
            break
          elif frame > 1000000:
            fvs.stop()
            break
      log.scene_recap(prev_scene)



