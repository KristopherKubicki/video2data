#!/usr/bin/python3 
# coding: utf-8
#
# Shot and scene printing

import datetime
from utils2 import image as v2dimage
from utils2 import sig as v2dsig
import titlecase

def shot_recap(prev_shot):
  shottime = prev_shot['shottime']
  adj = ''

  if prev_shot['end'] == prev_shot['start']:
    return

  #TODO: 
  #if prev_shot['break_count'] > 0:
  #  adj = 'Blank'

  # move to DA contrib. 
  #for rect in frame.get('object_rects'):
  #  if rect[2] in ['poster','billboard',' traffic sign'] and rect[1] > 0.2:
  #    adj = 'Information '
  #  if 'sign' in frame['summary']:
  #    adj = 'Information '

  print('\t[%s] Title: Unnamed %sShot %d' % (shottime,adj,prev_shot['start']))
  print('\t[%s] Length: %.02fs, %d frames, %d end' % (shottime,prev_shot['length'],prev_shot['end'] - prev_shot['start'],prev_shot['end']))
  print('\t[%s] Detect Method: %s ' % (shottime,prev_shot['break_type']))
  #print('\t[%s] Signature: %s' % (shottime,v2dsig.phash_bits(prev_shot['audioprint'])))
  if prev_shot['summary']:
    print('\t[%s] Description: %s' % (shottime,titlecase.titlecase(prev_shot['summary'])))
  if prev_shot['caption']:
    # TODO: this should really be shot_caption. frame needs to roll up to shot
    print('\t[%s] Caption: %s' % (shottime,prev_shot['caption']))
 # if frame['human_rects']:
 #   print('\t[%s] Humans: %s' % (shottime,len(prev_shot['human'])))
 # if shot_faces:
 #   print('\t[%s] Faces: %s' % (shottime,set(shot_faces)))
 # if text_rects:
 #   print('\t[%s] Words: %s' % (shottime,len(text_hulls)))
  print('\t[%s] Voice: %.1f%%' % (shottime,100*prev_shot['voiceprint'].count('1') / len(prev_shot['voiceprint'])))


def scene_recap(prev_scene):
  scenetime = prev_scene['scenetime']
  if prev_scene['end'] == prev_scene['start']:
    return

  print('[%s] Title: Unnamed %s' % (prev_scene['scenetime'],prev_scene.get('com_detect')))
  print('[%s] Length: %.02fs, %d frames' % (scenetime,prev_scene['length'],prev_scene['end'] - prev_scene['start']))
  print('[%s] Detect Method: %s ' % (scenetime,prev_scene['break_type']))  
  print('[%s] Signature: %s' % (scenetime,v2dsig.phash_bits(prev_scene['shotprint'])))
  print('[%s] CR %.1f%%' % (scenetime,100*prev_scene['shotprint'].count('1') / len(prev_scene['shotprint'])))

  if prev_scene['summary']:
    print('[%s] Description: %s' % (scenetime,titlecase.titlecase(prev_scene['summary'])))
  if prev_scene['caption']:
    print('[%s] Caption: %s' % (scenetime,prev_scene['caption']))
  if len(prev_scene['words']) > 0:
    print('[%s] Gist: %s' % (scenetime,set(prev_scene['words'])))
  if prev_scene['text']:
    print('[%s] Words: %s' % (scenetime,len(prev_scene['text'])))
  if len(prev_scene['faces']) > 0:
    print('[%s] Faces: %s' % (scenetime,set(prev_scene['faces'])))
  if len(prev_scene['stuff']) > 0:
    print('[%s] Objects: %d' % (scenetime,len(prev_scene['stuff'])))
  if prev_scene['vehicles']:
    print('[%s] Vehicles: %s' % (scenetime,len(prev_scene['vehicles'])))

