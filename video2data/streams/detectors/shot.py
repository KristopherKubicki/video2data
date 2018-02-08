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


  #if 'BLANK' in frame['shot_type']:
  #  frame['magic'] *= 3 
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
    frame['sbd'] *= 1.2
  elif frame['shot_detect'] == frame['frame'] and frame['magic'] > 2 and frame['shot_type'] == 'BLANK':
    #print('warning! magic blank shot!',frame['frame'],frame['audio_level'],frame['audio_max'])
    frame['scene_detect'] = frame['frame']
    frame['sbd'] *= 1.0

  # sometimes a scene will cut twice in rapid succession.  Give this an opportunity to trigger but
  #  not less than 1 second after it already cut
  #if frame['scene_detect'] == frame['frame'] and last_frame['scene_detect'] + frame['fps'] > frame['frame'] and frame['sbd'] < 1:
  #  print('warning, dropping double cut! ',frame['frame'],frame['sbd'])
  #  frame['scene_detect'] = last_frame['scene_detect']
  #  frame['sbd'] *= 0.1 
   
  # if we start getting ROS false positives, this is likely a highly isolated voice over commercial with no other audio
  #  the silence detector is very sensitive to this kind of audio, and thus the ROS detector has to be turned off if we 
  #  start getting misses on it.  It reactivates with a scene change, which means one of the other 10 backup mechanisms 
  #  have to find the break
  # Keep in mind, the ROS detector is responsible for almost all scene detection otherwise
  if frame['shot_detect'] != frame['frame'] and frame['audio_detect'] > frame['scene_detect'] + 3*frame['fps'] and frame['audio_detect'] < frame['frame'] - 5 and frame['isolation'] is False:
    #print(':::::::::::::  setting isolation!!!',frame['frame'])
    frame['isolation'] = True

  # its ok to unset the threshold if we are breaking scenes anyway
  #  we want to pick up candidates
  if frame['scene_detect'] > frame['frame'] - 20:
    frame['abd_thresh'] = 0

  if frame['scene_detect'] > frame['frame'] - 20 and frame['isolation'] is True:
    frame['isolation'] = False

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

  # broadcasters always go all the way to zero between shots.  Sometimes its for a very
  #  short time though. This works out to be about 15ms, with the windows being 1ms apart
  # TODO: Only do this if speech levels are low?
  #if frame['speech_level'] < 0.2:
  #
  # Audio between commercials tends to look like this
  #  #     #  #
  #  ##   #####
  #  ## # #####
  #  ##########
  #    ^ ^
  #  this feature is unique and pretty easy to spot.  
  #  sometimes its just a single trough
  #
  signs = np.sign(frame['audio_np'])
  signs[signs == 0] = -1
  frame['zcr'] = len(np.where(np.diff(signs))[0])/len(frame['audio_np']) + 0.0001

  signs = np.sign(frame['last_audio'])
  signs[signs == 0] = -1
  frame['last_zcr'] = len(np.where(np.diff(signs))[0])/len(frame['last_audio']) + 0.0001

  frame['ste'] = sum( [ abs(x)**2 for x in frame['audio_np'] ] ) / len(frame['audio_np'])
  last = frame['last_audio'][1::100]
  frame['last_ste'] = sum( [ abs(x)**2 for x in last ] ) / len(last)


  frame['rms'] = np.sqrt(np.mean(frame['audio_np']**2))
  frame['ster'] = frame['last_ste'] / frame['ste']
  frame['sterr'] = last_frame['ster'] / (frame['ster'] + 0.0001)
  frame['pwrr'] = frame['audio_power'] / frame['last_audio_power']
  # if zcr delta is high 
  frame['zcrr'] = frame['last_zcr'] / frame['zcr']
  frame['zcrrr'] = last_frame['zcrr'] / (frame['zcrr'] + 0.0001)
  #if frame['rms'] < 10: 
  #  print(' RMS! ',frame['frame'],frame['rms'],frame['audio_level'],frame['audio_mean'])

  # cheap, efficient energy detector. Anything over 0.01 is suspect, but not confirmed
  # the first occurence of this matters

  #mags1 = np.fft.rfft(frame['last_audio'])
  mags2 = np.fft.rfft(frame['audio_np'])
  mags = np.abs(mags2)
  amax = max(mags)
  amin = min(mags)
  variance = np.var(mags)
  frame['abd'] = (amax + amin) / variance

  audio_score = 1 
  audio_flag = ' '

  up_thresh = abs(frame['last_audio_mean']) - (abs(frame['last_audio_level']) / 2)
  std_mult = ((frame['last_audio_mean'] - frame['audio_mean']) / (frame['last_audio_level'] / 2))
  if std_mult > 2:
    print('std',std_mult,frame['last_audio_mean'],frame['audio_mean'],frame['last_audio_level']/2)
    #std_mult = 2

  # TODO:
  # there's a relationship between the labd and this 0.1
  #  the lower the labd, the higher the 0.1 should be
  #  
  # i calibrated against a mean of 500, with an adb of 0.001
  tmp1 = frame['last_audio_mean'] / 500
  #tmp2 = 0.001 / frame['last_abd']
  labd_ratio = 0.03 * tmp1
  #print('last_mean',frame['last_audio_mean'],labd_ratio)
  #print('tmp',tmp,frame['last_abd'],labd_ratio)
  last_mult = up_thresh * labd_ratio * std_mult

  frame['abd_thresh'] = last_frame['abd_thresh']
  # TODO: decay should be influenced by last average abd as well 
  frame['abd_thresh'] *= 0.995

  # theory: if we detect an anomaly but dont hit, that old anomaly level
  # becomes our new floor until the next scene break
  if (frame['abd'] > (last_frame['abd_thresh'] * 0.9) and frame['abd'] > 3 * (frame['last_abd'] / frame['magic'])):
    frame['abd_thresh'] = frame['abd']
    audio_flag = 'm%d' % last_mult
    #print('.. ',frame['frame'],last_mult)
    last_audio = np.fromstring(last_frame['audio'] + frame['audio'],np.int16)
    last_audio = abs(last_audio)

    # theory: the further the distance, the better likihood this is a break
    #   set the audio based on how big this distance is
    #print('\t\tGATE distance: %.3f' % std_mult,frame['abd_thresh'])
    #print('\t\tSTE GATE OPEN:',frame['frame'],frame['zcrr'],frame['ster'],frame['zcrrr'],frame['sterr'],last_mult,last_mult*0.2)
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
  elif std_mult > 2 and frame['magic'] >= 3:
    frame['audio_type'] = 'MAGIC1'
    frame['audio_detect'] = frame['frame']
  elif frame['sterr'] > 100 and frame['magic'] >= 3:
    frame['audio_type'] = 'MAGIC2'
    frame['audio_detect'] = frame['frame']
  elif frame['zcrrr'] > 100 and frame['magic'] >= 3:
    frame['audio_type'] = 'MAGIC3'
    frame['audio_detect'] = frame['frame']


#         data['last_audio'] = play_audio
#        data['last_audio_level'] = np.std(data['last_audio'])
#         data['last_audio_power'] = np.sum(data['last_audio']) / self.sample
#         data['last_rms'] = np.sqrt(np.mean(data['last_audio']**2))
#         data['last_audio_max'] = data['last_audio'].max(axis=0)
#         data['last_audio_mean'] = data['last_audio'].mean(axis=0)

   
  # new detectors 
  if frame['audio_detect'] == frame['frame']:
    audio_flag = 'a%d' % last_mult
  elif (frame['shot_detect'] == frame['frame']) and frame['magic'] > 1: 
    audio_flag = 'M%d' % last_mult
  elif audio_flag is ' ' and frame['abd'] > frame['last_abd'] * 5:
    audio_flag = 'D'
  #elif audio_flag is ' ' and frame['frame'] > 1250 and frame['frame'] < 1260:
  #  audio_flag = 'P'
  #elif audio_flag is ' ' and frame['frame'] > 2600 and frame['frame'] < 2615:
  #  audio_flag = 'P'


  if audio_flag is not ' ':
      #print(frame['frame'],': %.2f' % std_mult,'\t%s ' % audio_flag,frame['audio_type'],' score: ',audio_score,' abd: %.3f labd: %.3f trig: %.3f zcr %.2f zcrr %.2f zcrrr %.2f ster: %.2f '  % (frame['abd'],frame['last_abd'],frame['zcr'],frame['zcrr'],frame['zcrrr'],frame['ster']),' amax: %4d lmax: %4d ' % (frame['audio_max'], frame['last_audio_max']),' amean: %2d%%' % (100 * abs(frame['audio_mean']) / frame['last_audio_mean']),'shot: %s' % frame['shot_type'],'\tzeros: %4d' % frame['zeros'],' astd: %4d' % frame['audio_level'],' lstd: %d' % frame['last_audio_level'],' speech: %2d%%' % (frame['speech_level'] * 100),' arms: %2.1f' % frame['rms'],' lrms %2.1f' % frame['last_rms'], '  pwrr: %2.3f' % abs(frame['pwrr']))
      print(frame['frame'],': %.2f' % std_mult,'\t%s\t' % audio_flag,frame['audio_type'],'\tscore: ',audio_score,' abd: %.3f labd: %.3f trig: %.3f : zcr %.2f zcrr %.2f zcrrr %.2f ster: %.2f '  % (frame['abd'],frame['last_abd'],frame['abd_thresh'],frame['zcr'],frame['zcrr'],frame['zcrrr'],frame['ster']))
 

  return frame


