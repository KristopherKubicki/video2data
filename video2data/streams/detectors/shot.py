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

  if frame is None or last_frame is None:
    return frame

# copy the values from the last shot.  This gives us some short term memory 
  if last_frame and last_frame.get('shot_detect'):
    frame['shot_detect'] = last_frame['shot_detect']
  if last_frame and last_frame.get('audio_detect'):
    frame['audio_detect'] = last_frame['audio_detect']

#
# audio has to break in order to trigger a scene change
# potentially, I should only run this when the shot detect > frame - FPS
# 
  if frame.get('audio'):
    frame = audio_shot_detector(frame,last_frame)

  if last_frame.get('gray') is not None and frame.get('gray') is not None and last_frame.get('gray').shape == frame.get('gray').shape:
    frame = video_shot_detector(frame,last_frame)

# run the audio detector again if we believe we have a shot
  if frame['shot_detect'] == frame['frame'] and frame.get('audio') and frame.get('audio_type') is None:
    frame = audio_shot_detector(frame,last_frame)

  frame['break_type'] = '%s%s%d' % (frame['audio_type'],frame['shot_type'],frame['break_count'])


# TODO: Scene SVM here.  
#  Can a scene cut on motion?  I don't think so... 
#  but this seems to be true on ticker breaks
#  if 'MOTION' in frame['shot_type']:
#    return frame
  scene_magic = 2
  scene_len = int(10 * (frame['frame']-1 - frame['last_scene']) / frame['fps'])
  if scene_len % 50 in [0]:
    frame['sbd'] *= 3
    scene_magic = 4
  if scene_len % 50 in [49,1]:
    frame['sbd'] *= 1.5
    scene_magic = 3

# TODO
# Gray audio on a magic number

  if frame['audio_type'] and frame['shot_type']:
    print('HARD SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 2
  elif frame['audio_type'] and last_frame['shot_type'] and 'MOTION' not in last_frame['shot_type']:
    print('AUD SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.5
  elif last_frame['audio_type'] and frame['shot_type']:
    print('VID SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.5
  elif frame['shot_type'] and frame['audio_detect'] > frame['shot_detect'] - scene_magic:
    print('WA SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.2
  elif frame['audio_type'] and frame['shot_detect'] > frame['audio_detect'] - scene_magic:
    print('WV SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.2

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
    frame['shot_type'] = 'BLANK'
    frame['shot_detect'] = frame['frame']
    frame['sbd'] = 0.99
    #print('v\tBLANK', frame['frame'], frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))

  # FIXME: doesn't appear to be working. Check streams/read.py
  elif last_frame['width'] != frame['width'] or last_frame['height'] != frame['height']:
    frame['shot_type'] = 'RESIZE'
    frame['shot_detect'] = frame['frame']
    frame['sbd'] = 0.98
    #print('v\tRESIZE', frame['frame'], frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))

  # Histogram filter looks for a pretty large delta 
  #  
  elif abs(frame_mean - last_mean) > 5 and v2dimage.chi2_distance(frame['hist'],last_frame['hist']) > 1:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'HIST'
    frame['sbd'] = 0.9 + v2dimage.chi2_distance(frame['hist'],last_frame['hist']) / 100
    #print('v\tHIST',frame['frame'],frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))
    #cv2.imshow('Hist New',frame['small'])
    #cv2.imshow('Hist Old',last_frame['small'])
    #cv2.waitKey(1)
 
  # contrast filter checks to see if edges are changing
  elif abs(last_frame['contrast']- frame['contrast']) > 2 and abs(frame_mean - last_mean) > 0.5 and frame['mse'] > 1000:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'CONTRAST'
    frame['sbd'] = 0.8 + abs(last_frame['contrast']- frame['contrast']) / 100
    #print('v\tCONTRAST', len(frame['edges']), len(last_frame['edges']),frame['est_edge_delta'])

  elif abs(frame_mean - last_mean) > 2 and v2dimage.chi2_distance(frame['hist'],last_frame['hist']) > 0.5 and frame['gray_motion_mean'] > 5 and frame['mse'] > 1000:
    frame['shot_type'] = 'MOTION'
    frame['shot_detect'] = frame['frame']
    frame['motion_detect'] = frame['frame']
    frame['sbd'] = 0.7 + abs(frame_mean - last_mean) / 100
    #print('v MOTION', frame['frame'], frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))
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
     # print(frame['frame'],'v\t',frame['shot_type'],' %.1f %.1f :: ' % (frame['abd'],frame['zcr']),frame['frametime'],frame['frame'],'\t',frame['audio_type'],frame['break_count'],frame['silence'],frame['audio_level'],frame['speech_level'])
      print(frame['frame'],'v\t',frame['shot_type'],'%d:%d' % (last_frame.get('raudio_max'),frame.get('raudio_max')))
    frame['break_count'] += 1
  else:
    frame['break_count'] = 0

  return frame

def audio_shot_detector(frame,last_frame):

  if frame is None or last_frame is None:
    return frame

  # broadcasters always go all the way to zero between shots.  Sometimes its for a very
  #  short time though. This works out to be about 15ms, with the windows being 1ms apart
  #last_audio = np.fromstring(last_frame['audio'] + frame['audio'],np.int16)
  last_audio = np.fromstring(last_frame['audio'] + frame['audio'],np.int16)
  for i in range(23):
    frame['rolling_np'] = last_audio[i*100:i*100+735]
    if len(frame['rolling_np']) > 0:
      tmp = frame['rolling_np'].max(axis=0)
      if tmp < 3 and tmp < frame['raudio_max'] and tmp > 0:
        print('from',i*100,i*100+735,tmp,len(last_audio),len(frame['audio']))
        frame['raudio_max'] = frame['rolling_np'].max(axis=0)
        frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)

  # TODO: penalize blocks that falsly trigger this indicator
  #  Block ROS on first miss hit
  if frame['raudio_mean'] < 3:    
    print(frame['frame'],'ROLLING SILENCE!',frame['raudio_mean'],frame['raudio_max'])
  
  # cheap, efficient energy detector. Anything over 0.01 is suspect, but not confirmed
  # the first occurence of this matters
  #mags1 = np.fft.rfft(np.append(last_frame['audio_np'],frame['audio_np']))
  mags2 = np.fft.rfft(frame['audio_np'])
  s2 = set(mags2)
  #s1 = set(mags1)
  # indicates a signal is near zero 
  zero_crossings = np.where(np.diff(np.sign(mags2)))[0]
  mags = np.abs(mags2)
  amax = max(mags)
  amin = min(mags)
  variance = np.var(mags)
  frame['abd'] = (amax + amin) / variance
  frame['zcr'] = len(zero_crossings) / len(mags2)

  # check structure of audio (blocky, repetitive) 
  # check the audio level of the previous shot
  # check the shot cut rate.  "very stable"
  # TODO: implement ffmpeg loudnorm, otherwise these volume levels are arbitrary

  if frame['audio_level'] == 0:
    if frame['zcr'] > 0 and len(frame['edges']) == 0:
      frame['audio_type'] = 'HARD'
      frame['audio_detect'] = frame['frame']
    else:
      print('warning! audio defect')
  elif frame['raudio_mean'] < 5:
    frame['audio_type'] = 'ROS'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_mean'] < 0.1 and frame['audio_mean'] > -0.1 and frame['audio_max'] < 3:
    frame['audio_type'] = 'ZAA'
    frame['audio_detect'] = frame['frame']
  elif frame['raudio_mean'] < 0.1 and frame['raudio_mean'] > -0.1 and frame['raudio_max'] < 3:
    frame['audio_type'] = 'ZAR'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_max'] < 2:
    frame['audio_type'] = 'ZDB'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 1 and frame['silence'] > 50 and frame['speech_level'] < 0.2:
    frame['audio_type'] = 'ZEF'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 1 and frame['zcr'] > 0.4 and frame['speech_level'] < 0.2:
    frame['audio_type'] = 'ZRR'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_max'] < 10 and frame['silence'] > 500:
    frame['audio_type'] = 'ZZM'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_max'] < 10 and frame['silence'] > 500:
    frame['audio_type'] = 'ZZM'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 2 and frame['abd'] > 0.5 and frame['silence'] > 200 and last_frame['audio_detect'] == last_frame['frame']:
    frame['audio_type'] = 'CUT'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 20 and frame['abd'] > 0.01 and frame['speech_level'] < 0.1 and frame['audio_max'] < 10:
    frame['audio_type'] = 'PEAK'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 20 and frame['abd'] > 0.03 and frame['silence'] > 20 and frame['audio_max'] < 10:
    frame['audio_type'] = 'SOFT'
    frame['audio_detect'] = frame['frame']
#  elif frame['audio_level'] < 20 and frame['silence'] > 50 and frame['speech_level'] < 0.1:
#    frame['audio_type'] = 'MICRO'
#    frame['audio_detect'] = frame['frame']
#  elif frame['abd'] > 0.02 and frame['silence'] > 50 and frame['audio_level'] < 10:
#    frame['audio_type'] = 'CUT'
#    frame['audio_detect'] = frame['frame']
  elif frame['abd'] > 1:
    frame['audio_type'] = 'DEFECT'
    frame['audio_detect'] = frame['frame']
  elif frame['silence'] > 1000:
    frame['audio_type'] = 'NULL'
    frame['audio_detect'] = frame['frame']
   
  # blank + gray is usually a scene
   
  # new detectors 
  if frame['audio_detect'] == frame['frame']:
    print(frame['frame'],'a\t',frame['audio_type'],' %.3f %.2f :: ' % (frame['abd'],frame['zcr']),frame['frametime'],frame['audio_max'],frame['audio_mean'],'\t',frame['shot_type'],frame['break_count'],frame['silence'],frame['audio_level'],frame['speech_level'])
  elif frame['abd'] > 0.01 and frame['audio_level'] < 20 and frame['speech_level'] < 0.2:
    print(frame['frame'],'m\tGRAY\t\taudio %.3f %.2f :: ' % (frame['abd'],frame['zcr']),frame['audio_max'],frame['audio_mean'],'\t',frame['shot_type'],frame['break_count'],frame['silence'],frame['audio_level'])
  elif frame['zcr'] > 0.5 and frame['audio_level'] < 10:
    print(frame['frame'],'m\tZCR\t\taudio %.3f %.2f :: ' % (frame['abd'],frame['zcr']),frame['audio_max'],frame['audio_mean'],'\t',frame['shot_type'],frame['break_count'],frame['silence'],frame['audio_level'])
  elif frame['shot_detect'] == frame['frame']: 
    print(frame['frame'],'m\tMISS\taudio %.3f %.2f :: ' % (frame['abd'],frame['zcr']),frame['audio_max'],frame['audio_mean'],frame['frametime'],'\t',frame['shot_type'],frame['break_count'],frame['silence'],frame['audio_level'],frame['speech_level'])
  elif last_frame['audio_detect'] == frame['frame'] - 1:
    print(frame['frame'],'m\tECHO\t\taudio %.3f %.2f :: ' % (frame['abd'],frame['zcr']),frame['audio_max'],frame['audio_mean'],'\t',frame['shot_type'],frame['break_count'],frame['silence'],frame['audio_level'])
  elif frame['audio_mean'] < 0.1 and frame['audio_mean'] > -0.1 and frame['audio_max'] < 10:
    print(frame['frame'],'S\tSILENT\t\taudio %.3f %.2f :: ' % (frame['abd'],frame['zcr']),frame['audio_max'],frame['audio_mean'],'\t',frame['shot_type'],frame['break_count'],frame['silence'],frame['audio_level'])


 

  return frame


