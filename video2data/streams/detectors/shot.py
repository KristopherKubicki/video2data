#!/usr/bin/python3 
# coding: utf-8
#
import cv2
import numpy as np
from utils2 import image as v2dimage
import time

#
# Motion and Shot Detection. 
#  Creates a few masks that are used for underlying detectors.  Don't run NNs if there is no need...
#
def frame_to_contours(frame,last_frame):
  if frame is None or last_frame is None:
    return frame

  # A viewport is like an IFRAME for video
  # line detection with houghlines is too expensive.  we have to do the canny and the hough,
  #  both which we don't really need or want to do
  #  instead, naively check for rows that are the same color
  #  NOTE: this adds a few ms on the grayscale version.  its currently disabled 
  #  NOTE: this seems to work at higher resolutions better than lower, which hurts computation time a lot
  if frame['frame'] % 60 == 0 or last_frame['shot_detect'] == frame['frame'] - 1:
    start = time.time()
    tmp = frame['rframe']
    # CPU intensive on a big frame
    tmp = cv2.Canny(tmp,100,1)
    lines = cv2.HoughLinesP(tmp,1,np.pi/180,threshold=100,minLineLength=frame['height'] * 0.5,maxLineGap=0)
    tmp_frame = np.zeros((frame['height'],frame['width'],1),np.uint8)
    # push horizontal bars into array, and detect when those bars transit shots or scenes
    if lines is not None and len(lines):
      for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
          if y1 == y2 and x2 - x1 >= frame['width'] * 0.6:
            cv2.line(tmp_frame,(0,y1),(frame['width'],y1),(255),2)
          elif x1 == x2 and y1 - y2 >= frame['height'] * 0.5:
            cv2.line(tmp_frame,(x1,0),(x1,frame['height']),(255),2)
    #cv2.imshow('tmp',tmp_frame)
    _,cnts,hierarchy = cv2.findContours(cv2.bitwise_not(tmp_frame.copy()), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # TODO: sort cnts by area before selecting 
    frame['viewport'] = None
    for cnt in cnts:
      x,y,w,h = cv2.boundingRect(cnt)
      # TODO: check aspect ratio range
      if w > frame['width'] * 0.5 and h > frame['height'] * 0.5: 
        frame['viewport'] = cnt

  #  TODO: it's one thing to detect it, but its easier to track it
  #   consider following the box once it's been spotted
  else:
    frame['viewport'] = last_frame['viewport']

  # the viewport is only really useful to us if it is very stable
  if frame['viewport'] is not None:
    cv2.polylines(frame['show'],[frame['viewport']],True,(0,255,0),2)

  # TODO: detect disolves (contrast delta)
  #  blurs are also a way to get at that
  if last_frame.get('gray') is not None and frame.get('gray') is not None and last_frame.get('gray').shape == frame.get('gray').shape:
    # estimated gmm.  if necessary, do the slower calculation
    frame['gray_motion_mean'] = cv2.norm(last_frame['gray'], frame['gray'],cv2.NORM_L1) / float(frame['gray'].shape[0] * frame['gray'].shape[1])
    
    if frame['gray_motion_mean'] > 1:
      start = time.time() 
      frame_delta = cv2.absdiff(last_frame['gray'], frame['gray'])
      frame_delta = cv2.blur(frame_delta,(10,10))
      thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
      frame['gray_motion_mean'] = np.sum(thresh) / float(thresh.shape[0] * thresh.shape[1])
      motion_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
      frame['motion_hulls'] = [cv2.convexHull(p.reshape(-1, 1, 2) * int(1/frame['scale'])) for p in motion_cnts]
  return frame

def frame_to_shots(frame,last_frame):

  if frame is None or last_frame is None:
    return frame

# copy the values from the last shot.  This gives us some short term memory 
  if last_frame and last_frame.get('shot_detect'):
    frame['shot_detect'] = last_frame['shot_detect']
  if last_frame and last_frame.get('audio_detect'):
    frame['audio_detect'] = last_frame['audio_detect']
  if last_frame and last_frame.get('scene_detect'):
    frame['scene_detect'] = last_frame['scene_detect']
  if last_frame and last_frame.get('isolation'):
    frame['isolation'] = last_frame['isolation']

# TODO: Scene SVM here.  
#  Can a scene cut on motion?  I don't think so... 
#  but this seems to be true on ticker breaks
  frame['magic'] = 1
  scene_len = int(10 * (frame['frame'] - frame['last_scene']) / frame['fps'])
  if scene_len == 150 or scene_len == 300:
    frame['magic'] = 4
  elif scene_len >= 90 and scene_len % 50 in [0]:
    frame['magic'] = 3
  elif scene_len > 90 and scene_len % 50 in [49,1]:
    frame['magic'] = 2
  elif scene_len < 50:
    frame['magic'] = 0.5


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
  frame['sbd'] *= frame['magic']

  #
  # Allow the scene to cut if the audio and visual break at the same time, or within a 
  #  frame or two. An extra frame or two is given if the shot is aligning to a magic 5
  #  second interval
  #
  if frame['audio_type'] and frame['shot_type']:
    #print('HARD SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 2
  elif frame['audio_type'] and last_frame['shot_type']:
    #print('AUD SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.5
  elif last_frame['audio_type'] and frame['shot_type']:
    #print('VID SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.5
  elif frame['shot_type'] and frame['audio_detect'] >= frame['shot_detect'] - frame['magic'] - 1:
    #print('WA SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.2
  elif frame['audio_type'] and frame['shot_detect'] >= frame['audio_detect'] - frame['magic'] - 1:
    #print('WV SCENE',frame['frametime'],frame['frame'],'\t',frame['break_type'])
    frame['scene_detect'] = frame['frame']
    # FIXME: this is not proper, but we need to signal the shot here or the
    #  buffer might split it
    frame['shot_detect'] = frame['frame']
    frame['sbd'] *= 1.2

  # its ok to unset the threshold if we are breaking scenes anyway
  #  we want to pick up candidates
  if frame['scene_detect'] > frame['frame'] - 20:
    frame['abd_thresh'] = 0 
    frame['ster_thresh'] = 0

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
    #  print('v\tHIST',frame['frame'],frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))


  elif abs(last_frame['lum'] - frame['lum']) > 5 and v2dimage.chi2_distance(frame['hist'],last_frame['hist']) > 1:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'LUM'
    frame['sbd'] = 0.9 + v2dimage.chi2_distance(frame['hist'],last_frame['hist']) / 100
    #print('v\tLUM',frame['frame'],frame['lum'],frame['mse'],v2dimage.chi2_distance(frame['hist'],last_frame['hist']),frame['gray_motion_mean'],len(frame['motion_hulls']))
 
  # contrast filter checks to see if edges are changing
  elif abs(last_frame['contrast']- frame['contrast']) > 2 and abs(frame_mean - last_mean) > 0.5 and frame['mse'] > 1000:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'CONTRAST'
    frame['sbd'] = 0.8 + abs(last_frame['contrast']- frame['contrast']) / 100
    #print('v\tCONTRAST', len(frame['edges']), len(last_frame['edges']),frame['est_edge_delta'])

  elif frame['gray_motion_mean'] > 200 and frame['mse'] > 1000:
    frame['shot_detect'] = frame['frame']
    frame['shot_type'] = 'GMM'
    frame['sbd'] = 0.7 + (frame['gray_motion_mean'] / 100)
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
    #if frame['break_count'] == 0:
    #  print(frame['frame'],'v\t',frame['shot_type'],'%d:%d' % (last_frame.get('raudio_max'),frame.get('raudio_max')))
    frame['break_count'] += 1
  else:
    frame['break_count'] = 0

  return frame

def audio_shot_detector(frame,last_frame):

  frame['sterr'] = last_frame['ster'] / (frame['ster'] + 0.0001)
  frame['zcrrr'] = last_frame['zcrr'] / (frame['zcrr'] + 0.0001)

  audio_score = 1 
  audio_flag = ' '

  up_thresh = abs(frame['last_audio_mean']) - (abs(frame['last_audio_level']) / 2)
  std_mult = ((frame['last_audio_mean'] - frame['audio_mean']) / (frame['last_audio_level'] / 2))

  labd_ratio = 0.03 * frame['last_audio_mean'] / 500
  last_mult = up_thresh * labd_ratio * std_mult

  frame['abd_thresh'] = last_frame['abd_thresh']
  frame['abd_thresh'] *= 0.995

  frame['ster_thresh'] = last_frame['ster_thresh']
  frame['ster_thresh'] *= 0.995

  # theory: if we detect an anomaly but dont hit, that old anomaly level
  # becomes our new floor until the next scene break

  # TODO: check the ster component too
  if (frame['abd'] > last_frame['abd_thresh'] * 0.9 and frame['ster'] > last_frame['ster_thresh'] * 0.9):
    audio_flag = 'm%d' % last_mult
    #print('.. ',frame['frame'],last_mult)
    last_audio = np.fromstring(last_frame['audio'] + frame['audio'],np.int16)
    last_audio = abs(last_audio)

    for i in range(2):
      if audio_score > 1 or i*900+2048 > len(last_audio):
        break
      # small compromise on this part of the pyramid.
      frame['rolling_np'] = last_audio[i*900:i*900+2048]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= last_mult * 1 and rmax >= 0 and abs(rmean) <= last_mult * 1:
          if rmax < frame['raudio_max']:
            #print('>>>>>>>>> 9from',i,i*900,i*900+2048,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score =  9
    for i in range(4):
      if audio_score > 1 or i*512+1024 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*512:i*512+1024]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.9) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.9):
          if rmax < frame['raudio_max']:
            #print('>>>>>>>> 8from',i,i*512,i*512+1024,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score =  8
    for i in range(10):
      if audio_score > 1 or i*256+512 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*256:i*256+512]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.8) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.8):
          if rmax < frame['raudio_max']:
            #print('>>>>>> 7from',i,i*256,i*256+512,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score = 7
    for i in range(21):
      if audio_score > 1 or i*128+256 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*128:i*128+256]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.7) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.7):
          if rmax < frame['raudio_max']:
            #print('>>>>>> 6from',i,i*128,i*128+256,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score = 6
    for i in range(43):
      if audio_score > 1 or i*64+128 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*64:i*64+128]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.6) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.6):
          if rmax < frame['raudio_max']:
            #print('>>>>> 5from',i,i*64,i*64+128,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score = 5
    for i in range(89):
      if audio_score > 1 or i*32+64 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*32:i*32+64]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.5) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.5):
          if rmax < frame['raudio_max']:
            #print('>>>> 4from',i,i*32,i*32+64,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score = 4
    for i in range(187):
      if audio_score > 1 or i*16+32 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*16:i*16+32]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.4) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.4):
          if rmax < frame['raudio_max']:
            #print('>>> 3from',i,i*16,i*16+32,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score = 3
    for i in range(367):
      if audio_score > 1 or i*8+16 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*8:i*8+16]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.3) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.3):
          if rmax < frame['raudio_max']:
            #print('>> 2from',i,i*8,i*8+16,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score = 2  
    # this last guy really has to be zero, otherwise it will trip all the time
    #print('last_mult',last_mult,int(last_mult * 0.2))
    for i in range(734):
      if audio_score > 1 or i*4+8 > len(last_audio):
        break
      frame['rolling_np'] = last_audio[i*4:i*4+8]
      if len(frame['rolling_np']) > 0:
        rmax = frame['rolling_np'].max(axis=0)
        rmean = frame['rolling_np'].mean(axis=0)
        if rmax <= int(last_mult * 0.2) and rmax >= 0 and int(abs(rmean) <= last_mult * 0.2):
          if rmax < frame['raudio_max']:
            #print('> 1from',i,i*4,i*4+8,rmean,rmax,audio_score)
            frame['raudio_max'] = frame['rolling_np'].max(axis=0)
            frame['raudio_mean'] = frame['rolling_np'].mean(axis=0)
          audio_score = 1


  # check structure of audio (blocky, repetitive) 
  # TODO: 'dead' audio that isnt proceeded by a ROS might be corruption
  #  penalize, or make the scene capture rely heavily on video during defects
  if frame['zeros'] == len(frame['audio_np']):
    frame['audio_type'] = 'DEFECT'
    frame['audio_detect'] = frame['frame']
  elif frame['raudio_mean'] < 100:
    frame['audio_type'] = 'ROS%d' % audio_score 
    frame['audio_detect'] = frame['frame']
  elif frame['audio_mean'] < 0.1 and frame['audio_mean'] > -0.1 and frame['audio_max'] < 10 and last_frame['audio_type'][:3] == 'ROS':
    frame['audio_type'] = 'ROS+'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] == 0 and frame['zcr'] > 0:
    frame['audio_type'] = 'HARD'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_mean'] < 0.1 and frame['audio_mean'] > -0.1 and frame['audio_max'] < 3:
    frame['audio_type'] = 'ZAA'
    frame['audio_detect'] = frame['frame']
  elif frame['raudio_mean'] < 0.1 and frame['raudio_mean'] > -0.1 and frame['raudio_max'] < 3:
    frame['audio_type'] = 'ZAR'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_max'] < 2 and frame['rms'] < 10:
    frame['audio_type'] = 'ZDB'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 1 and frame['zeros'] > 50 and frame['speech_level'] < 0.2:
    frame['audio_type'] = 'ZEF'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 2 and frame['abd'] > 0.5 and frame['zeros'] > 200 and last_frame['audio_detect'] == last_frame['frame']:
    frame['audio_type'] = 'CUT'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 20 and frame['abd'] > 0.01 and frame['speech_level'] < 0.1 and frame['audio_max'] < 10:
    frame['audio_type'] = 'PEAK'
    frame['audio_detect'] = frame['frame']
  elif frame['audio_level'] < 20 and frame['abd'] > 0.03 and frame['zeros'] > 20 and frame['audio_max'] < 10:
    frame['audio_type'] = 'SOFT'
    frame['audio_detect'] = frame['frame']
  elif frame['zeros'] > len(frame['audio_np']) * 0.9:
    frame['audio_type'] = 'NULL'
    frame['audio_detect'] = frame['frame']
# magic settings
  elif std_mult > 2 and frame['magic'] >= 3:
    frame['audio_type'] = 'MAGIC1'
    frame['audio_detect'] = frame['frame']
  elif frame['sterr'] > 100 and frame['magic'] >= 3:
    frame['audio_type'] = 'MAGIC2'
    frame['audio_detect'] = frame['frame']
  elif frame['zcrrr'] > 100 and frame['magic'] >= 3:
    frame['audio_type'] = 'MAGIC3'
    frame['audio_detect'] = frame['frame']
  elif frame['sterr'] > 1000 and frame['magic'] >= 2:
    frame['audio_type'] = 'MAGIC4'
    frame['audio_detect'] = frame['frame']
  elif (frame['shot_detect'] == frame['frame']) and frame['magic'] > 1:
    print(frame['frame'],': %.2f' % std_mult,'\t%s\t' % audio_flag,frame['audio_type'],'\tscore: ',audio_score,' abd: %.3f labd: %.3f trig: %.3f : zcr %.2f zcrr %.2f zcrrr %.2f ster: %.2f '  % (frame['abd'],frame['last_abd'],frame['abd_thresh'],frame['zcr'],frame['zcrr'],frame['zcrrr'],frame['ster']))
    #frame['audio_type'] = 'MAGIC5'
    #frame['audio_detect'] = frame['frame']
    audio_flag = 'M%d' % last_mult

  # new detectors 
  #if frame['audio_detect'] == frame['frame']:
  #  audio_flag = 'a%d' % last_mult
  #elif audio_flag is ' ' and frame['abd'] > frame['last_abd'] * 5 and (frame['ster'] > frame['mean_ste'] * 10 or frame['ster'] < frame['mean_ste'] * 0.1):
  #  audio_flag = 'S%d' % last_mult
  #elif (frame['shot_detect'] == frame['frame']) and frame['magic'] > 1: 
  #  audio_flag = 'M%d' % last_mult
  #elif audio_flag is ' ' and frame['abd'] > frame['last_abd'] * 10:
  #  audio_flag = 'D'
  #elif audio_flag is ' ' and frame['frame'] > 1250 and frame['frame'] < 1260:
  #  audio_flag = 'P'
  #elif audio_flag is ' ' and frame['frame'] > 2600 and frame['frame'] < 2615:
  #  audio_flag = 'P'
  #
  #if audio_flag is not ' ' or frame['audio_type'] or frame['frame'] in range(590,602):
  #  print(frame['frame'],': %.2f' % std_mult,'\t%s\t' % audio_flag,frame['audio_type'],'\tscore: ',audio_score,' abd: %.3f : zcr %.2f : ster: %.3f'  % (frame['abd'] / frame['abd_thresh'],frame['zcr'],frame['ster'] / frame['ster_thresh']))
  #  print(frame['frame'],': %.2f' % std_mult,'\t%s\t' % audio_flag,frame['audio_type'],'\tscore: ',audio_score,' abd: %.3f labd: %.3f trig: %.3f : zcr %.2f zcrr %.2f zcrrr %.2f ster: %.2f '  % (frame['abd'],frame['last_abd'],frame['abd_thresh'],frame['zcr'],frame['zcrr'],frame['zcrrr'],frame['ster']))
 
  if (frame['abd'] > last_frame['abd_thresh'] * 0.9 and frame['ster'] > last_frame['ster_thresh'] * 0.9):
    frame['abd_thresh'] = frame['abd']
    frame['ster_thresh'] = frame['ster']

  return frame


