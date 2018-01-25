#!/usr/bin/python3 
# coding: utf-8
#
# Shot and scene printing

import datetime
from utils2 import image as v2dimage
import titlecase

def shot_recap(frame):
  frame_type = '%s%s%d' % (audio_type,shot_type,break_count)
  if shot_detect == frame:
    shottime = datetime.datetime.utcnow().utcfromtimestamp((last_shot + break_count) / FPS).strftime('%H:%M:%S')
    adj = ''
    for rect in oi_rects:
      if rect[2] in ['poster','billboard',' traffic sign'] and rect[1] > 0.2:
        adj = 'Information '
      if 'sign' in data['shot_summary']:
        adj = 'Information '

    print('\t[%s] Title: Unnamed %sShot %d' % (shottime,adj,frame))
    print('\t[%s] Length: %.02fs, %d frames' % (shottime,float((frame - last_shot + break_count) / FPS),frame-last_shot + break_count))
    #print('\t[%s] Detect Method: %s ' % (shottime,frame_type))
    print('\t[%s] Signature: %s' % (shottime,v2dsig.phash_bits(shot_audioprint)))
    if data['shot_summary']:
      print('\t[%s] Description: %s' % (shottime,titlecase.titlecase(data['shot_summary'])))
    if prev_scene['caption']:
      # TODO: this should really be shot_caption. frame needs to roll up to shot
      print('\t[%s] Caption: %s' % (shottime,frame_caption))
    if frame['human_rects']:
        print('\t[%s] Humans: %s' % (shottime,len(frame['human_rects'])))
    if shot_faces:
      print('\t[%s] Faces: %s' % (shottime,set(shot_faces)))
    if text_rects:
      print('\t[%s] Words: %s' % (shottime,len(text_hulls)))
    print('\t[%s] Voice: %.1f%%' % (shottime,100*shot_voiceprint.count('1') / len(shot_voiceprint)))
    return frame

def scene_recap(prev_scene):
  scenetime = prev_scene['scenetime']
  print('[%s] Title: Unnamed %s' % (prev_scene['scenetime'],prev_scene['com_detect']))
  print('[%s] Length: %.02fs, %d frames' % (scenetime,prev_scene['length'],frame - last_scene + break_count))
  #print('[%s] Detect Method: %s ' % (filetime,prev_scene['detect']))  
  print('[%s] Signature: %s' % (scenetime,v2dsig.phash_bits(prev_scene['scene_shotprint'])))
  if prev_scene['shot_summary']:
    print('[%s] Description: %s' % (scenetime,titlecase.titlecase(prev_scene['shot_summary'])))
  if prev_scene['caption']:
    print('[%s] Caption: %s' % (scenetime,prev_scene['caption']))
  if len(prev_scene['scene_words']) > 0:
    print('[%s] Gist: %s' % (scenetime,set(prev_scene['scene_words'])))
  if prev_scene['scene_text']:
    print('[%s] Words: %s' % (scenetime,len(prev_scene['scene_text'])))
  if len(prev_scene['scene_faces']) > 0:
    print('[%s] Faces: %s' % (scenetime,set(prev_scene['scene_faces'])))
  if len(prev_scene['scene_stuff']) > 0:
    print('[%s] Objects: %d' % (scenetime,len(prev_scene['scene_stuff'])))
  if prev_scene['scene_vehicles']:
    print('[%s] Vehicles: %s' % (scenetime,len(prev_scene['scene_vehicles'])))

