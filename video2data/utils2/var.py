#!/usr/bin/python3 
# coding: utf-8

def new_scene(frame_num=0):
  prev_scene = {}
  prev_scene['type'] = 'Segment'
  prev_scene['length'] = 0
  prev_scene['caption'] = None
  prev_scene['opening'] = None
  prev_scene['closing'] = None
  prev_scene['shotprint'] = '0b'
  prev_scene['text'] = []
  prev_scene['summary'] = ''
  prev_scene['words'] = ''
  prev_scene['start'] = frame_num
  prev_scene['faces'] = []
  prev_scene['vehicles'] = []
  prev_scene['people'] = []
  prev_scene['stuff'] = []
  prev_scene['objects'] = []
  prev_scene['break_type'] = ''
  return prev_scene

def new_shot(frame_num=0):
  prev_shot = {}
  prev_shot['start'] = frame_num
  prev_shot['type'] = 'Segment'
  prev_shot['method'] = ''
  prev_shot['length'] = 0
  prev_shot['caption'] = None
  prev_shot['closing'] = None
  prev_shot['text'] = []
  prev_shot['faces'] = []
  prev_shot['vehicles'] = []
  prev_shot['objects'] = []
  prev_shot['audioprint'] = '0b'
  prev_shot['textprint'] = '0b'
  prev_shot['voiceprint'] = '0b'
  prev_shot['motionprint'] = '0b'
  prev_shot['faceprint'] = '0b'
  prev_shot['summary'] = ''
  prev_shot['break_type'] = ''
  return prev_shot

