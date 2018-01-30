#!/usr/bin/python3 
# coding: utf-8
#
import cv2
import numpy as np
from utils2 import image as v2dimage
import time

#mser = cv2.MSER_create()

#
# Motion and Shot Detection. 
#  Creates a few masks that are used for underlying detectors.  Don't run NNs if there is no need...
#
def frame_to_contours(frame,last_frame):
  if frame is None or last_frame is None:
    return frame

  # TODO: detect disolves (contrast delta)
  #  blurs are also a way to get at that
  if last_frame.get('gray') is not None and frame.get('gray') is not None and last_frame.get('gray').shape == frame.get('gray').shape:
    frame['gray_motion_mean'] = cv2.norm(last_frame['gray'], frame['gray'],cv2.NORM_L1) / float(frame['gray'].shape[0] * frame['gray'].shape[1])
    
    if frame['gray_motion_mean'] > 1:
      start = time.time() 
      frame_delta = cv2.absdiff(last_frame['gray'], frame['gray'])
      frame_delta = cv2.blur(frame_delta,(10,10))
      thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
      frame['gray_motion_mean'] = np.sum(thresh) / float(thresh.shape[0] * thresh.shape[1])
      motion_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
      frame['motion_hulls'] = [cv2.convexHull(p.reshape(-1, 1, 2) * int(1/frame['scale'])) for p in motion_cnts]
    
    # potentially compute intensive
    #tmp = cv2.Canny(frame['gray'],10,200)
    #frame['edges'] = cv2.findContours(tmp.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    #cv2.imshow('edges',tmp)
  return frame

def frame_to_shots(frame,last_frame):
  frame['shot_detect'] = 0
  if last_frame and last_frame.get('shot_detect'):
    frame['shot_detect'] = last_frame['shot_detect']

  if frame is None or last_frame is None:
    return frame

  if last_frame.get('gray') is not None and frame.get('gray') is not None and last_frame.get('gray').shape == frame.get('gray').shape:
    frame = video_shot_detector(frame,last_frame)

# audio has to break in order to trigger a scene change
# potentially, I should only run this when the shot detect > frame - FPS
# 
# TODO:  Implement this across the play_audio stream as well 
#
  if frame.get('audio'):
    frame = audio_shot_detector(frame,last_frame)

  frame['break_type'] = '%s%s%d' % (frame['audio_type'],frame['shot_type'],frame['break_count'])
  if last_frame['audio_type'] and frame['audio_type'] is '' and frame['shot_type']:
    frame['audio_type'] = last_frame['audio_type']

  if frame['audio_type'] and frame['shot_type']:
#    if last_frame['scene_detect'] != frame['frame'] - 1:
    print('SCENE!',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 2

  return frame


def video_shot_detector(frame,last_frame):


  frame['mse'] = np.sum((frame['gray'].astype("float") - last_frame['gray'].astype("float")) ** 2)
  frame['mse'] /= float(frame['gray'].shape[0] * frame['gray'].shape[1])

  #regions,_ = mser.detectRegions(frame['gray'])
  #hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
  #cv2.polylines(frame['show'], hulls, 1, (0, 255, 0))
  #print('mse/mser',frame['mse'],len(hulls))

  # cheap and easy helper delta for dissolves
  frame['est_edge_delta'] = 0
  frame['edges'] = []
  #if last_frame.get('edges') is not None and frame.get('edges') is not None:
  #  if len(frame['edges']) == 0:
  #    frame['est_edge_delta'] = 1
  #  else:
  #    frame['est_edge_delta'] = abs(len(frame['edges']) - len(last_frame['edges'])) / len(frame['edges'])

  frame['break_count'] = last_frame.get('break_count')
  if frame['break_count'] is None:
    frame['break_count'] = 0

  frame['scene_detect'] = last_frame.get('scene_detect')
  if frame['scene_detect'] is None:
    frame['scene_detect'] = 0

  frame_mean = frame['frame_mean']
  last_mean = last_frame['frame_mean']
  if frame_mean < 5 or frame_mean > 250 and len(frame['edges']) == 0:
    frame['shot_type'] = 'Blank'
    frame['shot_detect'] = frame['frame']
    frame['sbd'] = 0.99
    print('BLANK', frame['frame'], frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))
  elif last_frame['width'] != frame['width'] or last_frame['height'] != frame['height']:
    frame['shot_type'] = 'Resize'
    frame['shot_detect'] = frame['frame']
    frame['sbd'] = 0.98
    print('RESIZE', frame['frame'], frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))

  # Histogram filter looks for a pretty large delta 
  #  
  elif abs(frame_mean - last_mean) > 5 and v2dimage.chi2_distance(frame['hist'],last_frame['hist']) > 1:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'Hist'
    frame['sbd'] = 0.9 + v2dimage.chi2_distance(frame['hist'],last_frame['hist']) / 100
    print('HIST', frame['frame'], frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))
 
  # contrast filter checks to see if edges are changing
  elif abs(last_frame['contrast']- frame['contrast']) > 2 and abs(frame_mean - last_mean) > 0.5 and frame['mse'] > 1000:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'Contrast'
    frame['sbd'] = 0.8 + abs(last_frame['contrast']- frame['contrast']) / 100
    print('CONTRAST', len(frame['edges']), len(last_frame['edges']),frame['est_edge_delta'])

  elif abs(frame_mean - last_mean) > 2 and v2dimage.chi2_distance(frame['hist'],last_frame['hist']) > 0.5 and frame['gray_motion_mean'] > 5 and frame['mse'] > 1000:
    frame['shot_type'] = 'Motion'
    frame['shot_detect'] = frame['frame']
    frame['motion_detect'] = frame['frame']
    frame['sbd'] = 0.7 + abs(frame_mean - last_mean) / 100
    print('MOTION', frame['frame'], frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))
  elif len(frame['motion_hulls']) > 0:
    frame['motion_detect'] = frame['frame']
    frame['sbd'] = 0.1
  else:
    frame['sbd'] = 0.02

  #if last_frame.get('edges') is not None:
  #  print('EDGES', frame['frame'], len(frame['edges']), len(last_frame['edges']),frame['est_edge_delta'])

  if frame['motion_detect'] and last_frame['motion_detect'] != frame['frame'] - 1:
    frame['motion_start'] = frame['motion_detect']
  if frame['shot_detect'] == frame['frame']:
    frame['motion_detect'] = frame['frame']
    if frame['break_count'] == 0:
      print('\tvideo',frame['frametime'],frame['frame'],'\t',frame['shot_type'],frame['break_count'])
    frame['break_count'] += 1
  else:
    frame['break_count'] = 0

  return frame

def audio_shot_detector(frame,last_frame):

  frame['audio_type'] = ''
  frame['audio_detect'] = 0
  if frame is None or last_frame is None:
    return frame
  
  # cheap, efficient energy detector. Anything over 0.01 is suspect, but not confirmed
  # the first occurence of this matters
  #tmp = last_frame['audio_np'].copy()
  #tmp.append(frame['audio_np'])
  #mags1 = np.fft.rfft(np.append(last_frame['audio_np'],frame['audio_np']))
  mags2 = np.fft.rfft(frame['audio_np'])
  s2 = set(mags2)
  # indicates a signal is near zero 
  zero_crossings = np.where(np.diff(np.sign(mags2)))[0]
  mags = np.abs(mags2)
  amax = max(mags)
  amin = min(mags)
  variance = np.var(mags)
  score = (amax + amin) / variance

  # check structure of audio (blocky, repetitive) 
  # check the audio level of the previous shot
  # check the shot cut rate.  "very stable"


  if frame['audio_level'] == 0:
    if len(frame['edges']) == 0:
      frame['audio_type'] = 'Hard'
      frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 20 and score > 0.01:
    frame['audio_type'] = 'Peak'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 1 and frame['silence'] > 10:
    frame['audio_type'] = 'Soft'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 20 and frame['silence'] > 50:
    frame['audio_type'] = 'Micro'
    frame['audio_detect'] = frame['frame']
  elif score > 1:
    frame['audio_type'] = 'Defect'
    frame['audio_detect'] = frame['frame']
  #elif frame['audio_level'] < 10 and frame['silence'] > 5:
  #  frame['audio_type'] = 'Gray'
  #  frame['audio_detect'] = frame['frame']
  #elif frame['audio_level'] < 20 and frame['silence'] > 10:
  #  frame['audio_type'] = 'Micro'
  #  frame['audio_detect'] = frame['frame']
   
 
  zcr = len(zero_crossings) / len(mags2)
  if frame['audio_detect'] == frame['frame']:
    print('\tHIT\taudio %.1f %.1f :: ' % (score * 1000,10 * zcr),len(s2),frame['frametime'],frame['frame'],'\t',frame['audio_type'],frame['break_count'],frame['silence'],frame['audio_level'],frame['speech_level'])
  elif frame['shot_detect'] == frame['frame']: 
    print('\tMISS\taudio %.1f %.1f :: ' % (score * 1000,10 * zcr),len(s2),frame['frametime'],frame['frame'],'\t',frame['audio_type'],frame['break_count'],frame['silence'],frame['audio_level'],frame['speech_level'])

  if score > 0.01:
    print('\tGRAY1\t\taudio %.1f %.1f :: ' % (score * 1000,10 * zcr),len(s2),frame['frame'],'\t',frame['audio_type'],frame['break_count'],frame['silence'],frame['audio_level'])
  elif zcr > 0.5:
    print('\tGRAY2\t\taudio %.1f %.1f :: ' % (score * 1000,10 * zcr),len(s2),frame['frame'],'\t',frame['audio_type'],frame['break_count'],frame['silence'],frame['audio_level'])

  return frame


