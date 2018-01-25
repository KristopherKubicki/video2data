#!/usr/bin/python3 
# coding: utf-8
#
import cv2
import numpy as np
from utils2 import image as v2dimage

#
# Motion and Shot Detection. 
#  Creates a few masks that are used for underlying detectors.  Don't run NNs if there is no need...
#
def frame_to_contours(frame,last_frame):
  if frame is None or last_frame is None:
    return frame

  # TODO: detect disolves (contrast delta)
  #  blurs are also a way to get at that
  frame['shot_type'] = ''
  if last_frame.get('gray') is not None and frame.get('gray') is not None and last_frame.get('gray').shape == frame.get('gray').shape:
    h1,w1 = last_frame.get('gray').shape
    frame_delta = cv2.absdiff(last_frame['gray'], frame['gray'])
    frame_delta = cv2.blur(frame_delta,(10,10))
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    motion_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    frame['motion_hulls'] = [cv2.convexHull(p.reshape(-1, 1, 2) * int(1/frame['scale'])) for p in motion_cnts]
    frame['gray_motion_mean'] = np.sum(thresh) / float(thresh.shape[0] * thresh.shape[1])
    frame['edges'] = cv2.Canny(frame['gray'],100,200)
  return frame

def frame_to_shots(frame,last_frame):
  frame['shot_detect'] = 0
  if last_frame and last_frame.get('shot_detect'):
    frame['shot_detect'] = last_frame['shot_detect']
  frame['motion_detect'] = 0
  frame['scene_detect'] = 0
  frame['audio_type'] = ''
  frame['shot_type'] = ''
  frame['break_count'] = 0

  if frame is None or last_frame is None:
    return frame

  if last_frame.get('gray') is not None and frame.get('gray') is not None and last_frame.get('gray').shape == frame.get('gray').shape:
    frame = video_shot_detector(frame,last_frame)

# audio has to break in order to trigger a scene change
# potentially, I should only run this when the shot detect > frame - FPS
  if frame.get('audio'):
    frame = audio_shot_detector(frame,last_frame)

  frame['frame_type'] = '%s%s%d' % (frame['audio_type'],frame['shot_type'],frame['break_count'])
  return frame


def video_shot_detector(frame,last_frame):

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
  elif last_frame['width'] != frame['width'] or last_frame['height'] != frame['height']:
    frame['shot_type'] = 'Resize'
    frame['shot_detect'] = frame['frame']
  elif abs(frame_mean - last_mean) > 5 and v2dimage.chi2_distance(frame['hist'],last_frame['hist']) > 1:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'Hist'
  elif abs(last_frame['contrast']- frame['contrast']) > 2 and abs(frame_mean - last_mean) > 0.5:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'Contrast'
  elif abs(frame_mean - last_mean) > 2 and v2dimage.chi2_distance(frame['hist'],last_frame['hist']) > 0.5 and frame['frame'] > frame['shot_detect'] + 5 and frame['gray_motion_mean'] > 5:
    frame['shot_type'] = 'Motion'
    frame['shot_detect'] = frame['frame']
    frame['motion_detect'] = frame['frame']
  elif len(frame['motion_hulls']) > 0:
    frame['motion_detect'] = frame['frame']

  if frame['motion_detect'] and last_frame['motion_detect'] != frame['frame'] - 1:
    frame['motion_start'] = frame['motion_detect']
  if frame['shot_detect'] == frame['frame']:
    frame['motion_detect'] = frame['frame']
    frame['break_count'] += 1

  return frame


def audio_shot_detector(frame,last_frame):

  frame['audio_type'] = ''
  frame['audio_detect'] = 0
  if frame is None or last_frame is None:
    return frame

  scene_length = '%d' % int((frame['frame'] - last_frame['scene_detect'] + frame['break_count']) / frame['fps'])
  if frame['audio_level'] == 0:
    # don't do this if there are contours on the screen!
    if len(frame['edges']) == 0 or scene_length in [5,10,15,30,45,60,90,120]:
      frame['audio_type'] = 'Hard'
      frame['audio_detect'] = frame['frame']
    elif frame['audio_level'] < 10 and scene_length in [5,10,15,30,45,60,90,120]:
      frame['audio_type'] = 'Soft'
      frame['audio_detect'] = frame['frame']
    elif frame['audio_level'] < 20 and frame['silence'] > 5:
      frame['audio_type'] = 'Gray'
      frame['audio_detect'] = frame['frame']
    elif frame['audio_level'] < 30 and frame['silence'] > 10:
      frame['audio_type'] = 'Micro'
      frame['audio_detect'] = frame['frame']

  return frame


