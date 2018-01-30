#!/usr/bin/python3 
# coding: utf-8
 
import os, sys
import random
import threading
import multiprocessing
import subprocess
import streams.transcribe
import streams.transcribe
import queue
import select
import time
from fcntl import fcntl, F_GETFL, F_SETFL, ioctl
import cv2
import numpy as np
import re
import select

from streams.detectors import shot

# TODO:
#  structure this more intelligently
def new_frame(frame_num=0):
  frame = {}
  frame['frame'] = frame_num
  # TODO: initialize this properly
  frame['frametime'] = ''

  # TODO: initialize these properly
  #frame['small'] = bytes()
  #frame['gray'] = bytes()
  #frame['rframe'] = bytes()

  # dictates what kind of shot boundary detected
  frame['shot_type'] = ''
  # edge contours, good for rough estimations
  frame['edges'] = []
  # proprietary metric 
  frame['gray_motion_mean'] = 0
  # another video break metric
  frame['mse'] = 0
  # shot break detection score
  frame['sbd'] = 0.0
  # hulls from motion
  frame['motion_hulls'] = []

  # last shot detected, could be itself
  frame['shot_detect'] = 0
  # last motion frame detected, could be itself
  frame['motion_detect'] = 0
  # last audio break detected, could be itself
  frame['audio_detect'] = 0
  # last scene break detected, could be itself
  frame['scene_detect'] = 0
  # type of audio detected
  frame['audio_type'] = ''
  # how long has this frame been breaking?
  frame['break_count'] = 0
  # zeros in the audio frame - a measure of artificial silence
  frame['silence'] = 0

  # how much speech is in the frame
  frame['speech_level'] = 0.0
  return frame

#
# given a segment of audio, determine its energy
#  check if that energy is in the human band of 
#  speech.  
#
#  warning - this has a defect in it that will crash for long
#  segments of audio.  Needs to be fixed
#
def speech_calc(self,raw_audio):
  if len(raw_audio) < 10:
    return 0

  data_freq = np.fft.fftfreq(len(raw_audio),1.0/self.sample)
  data_freq = data_freq[1:]
  data_ampl = np.abs(np.fft.fft(raw_audio))
  data_ampl = data_ampl[1:]
  data_energy = data_ampl ** 2

  energy_freq = {}
  for (i, freq) in enumerate(data_freq):
    if abs(freq) not in energy_freq:
      energy_freq[abs(freq)] = data_energy[i] * 2
  sum_energy = 0
  for f in energy_freq.keys():
    if 300<f<3000:
      sum_energy += energy_freq[f]
  if sum_energy < 1:
    return 0

  return sum_energy / sum(energy_freq.values())

# 
# This is the main thread that processes ffmpeg,
# and muxes everything together out of the fifos
#
class FileVideoStream:

  def process(self):

    if self.image:
      return

      #
      # self.pipe is the pipe from the ffmpeg subprocess
      # self.video_fifo is the video channel from the ffmpeg proces
      # self.audio_fifo is the audio channel from the ffmpeg proces
      # self.caption_fifo is the caption channel from the ffmpeg proces 
      #   captions are pulled from a filter as they are embedded in the 
      #   video meta data and otherwise lost on the transcode
      #
    if self.pipe is not None:
      self.pipe.terminate()
      #self.pipe.kill()
    if self.video_fifo is not None:
      os.close(self.video_fifo)
    if self.audio_fifo is not None:
      os.close(audio_fifo)
    if self.caption_fifo is not None:
      os.close(caption_fifo)

    # not thread safe
    if os.path.exists('/tmp/%s_video' % self.name):
      os.unlink('/tmp/%s_video' % self.name)
    if os.path.exists('/tmp/%s_audio' % self.name):
      os.unlink('/tmp/%s_audio' % self.name)
    if os.path.exists('/tmp/%s_caption' % self.name):
      os.unlink('/tmp/%s_caption' % self.name)

    os.mkfifo('/tmp/%s_video' % self.name)
    os.mkfifo('/tmp/%s_audio' % self.name)
    os.mkfifo('/tmp/%s_caption' % self.name)
 
    video_command = []
    #
    #  big assumption: http stream is an MPEGTS and probably a TV source
    # TODO: ffmpeg command passthrough.  enable / disable logging
    # 
    if self.stream[:4] != 'rtsp':
      # probably should do this better
      self.stream = str.replace(self.stream,':','\\\:')
      self.stream = str.replace(self.stream,'@','\\\@')
 
      # we should probably always stick to 're' for most video
      video_command = [ 'ffmpeg','-nostdin','-hide_banner','-loglevel','panic','-hwaccel','vdpau','-y','-f', 'lavfi','-i','movie=%s:s=0\\\:v+1\\\:a[out0+subcc][out1]' % self.stream,'-map','0:v','-vf','scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2' % (self.width,self.height,self.width,self.height),'-pix_fmt','bgr24','-r','%f' % self.fps,'-s','%dx%d' % (self.width,self.height),'-vcodec','rawvideo','-f','rawvideo','/tmp/%s_video' % self.name, '-map','0:a','-acodec','pcm_s16le','-r','%f' % self.fps,'-ab','16k','-ar','%d' % self.sample,'-ac','1','-f','wav','/tmp/%s_audio' % self.name, '-map','0:s','-f','srt','/tmp/%s_caption' % self.name  ] 
      self.pipe = subprocess.Popen(video_command)

      print('Step 2 initializing video /tmp/%s_video' % self.name)
      if not os.path.exists('/tmp/%s_video' % self.name):
        return
      self.video_fifo = os.open('/tmp/%s_video' % self.name,os.O_RDONLY | os.O_NONBLOCK,self.width*self.height*3*10)
      #fcntl(self.video_fifo,1031,6220800)
      fcntl(self.video_fifo,1031,1048576)

      print('Step 3 initializing video /tmp/%s_video' % self.name)
      self.audio_fifo = os.open('/tmp/%s_audio' % self.name,os.O_RDONLY | os.O_NONBLOCK,self.sample*2*10)
      fcntl(self.audio_fifo,1031,1048576)
      #fcntl(self.audio_fifo,1031,6220800)

      # TODO: move this to os.open()
      #self.caption_fifo = open('/tmp/%s_caption' % self.name,'rb',4096)
      self.caption_fifo = os.open('/tmp/%s_caption' % self.name,os.O_RDONLY | os.O_NONBLOCK,4096)
      #flags = fcntl(self.caption_fifo, F_GETFL)
      #fcntl(self.caption_fifo, F_SETFL, flags | os.O_NONBLOCK)
    #
    # big assumption rtsp stream is h264 of some kind and probably a security camera
    #  it does not handle audio, as the audio not actually in the stream.  Presumably we could mux it here if it exists
    elif self.stream[:4] == 'rtsp':
      # we don't need the lavfi command because there are no subs.  also, the movie= container for rtsp doesn't let me pass rtsp_transport, which will result in
      #  dropped packets if I do not do
      # also, vdpau is not working for some reason (yuv issues)
      # 
      video_command = [ 'ffmpeg','-nostdin','-re','-hide_banner','-loglevel','panic','-y','-r','%f' % self.fps,'-rtsp_transport','tcp','-i',self.stream,'-map','0:v','-vf','scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2' % (self.width,self.height,self.width,self.height),'-pix_fmt','bgr24','-r','%f' % self.fps,'-vcodec','rawvideo','-f','rawvideo','/tmp/%s_video' % self.name ]

      self.pipe = subprocess.Popen(video_command)
      print('fifo: /tmp/%s_video' %self.name)
      self.video_fifo = os.open('/tmp/%s_video' % self.name,os.O_RDONLY | os.O_NONBLOCK,self.width*self.height*3*10)
    else:
      print("unrecognized input")
      return
 
    self.audio_poll = None
    self.video_poll = None
    self.caption_poll = None
    if self.caption_fifo:
      self.caption_poll = select.poll()
      self.caption_poll.register(self.caption_fifo,select.POLLIN)
    if self.video_fifo:
      self.video_poll = select.poll()
      self.video_poll.register(self.video_fifo,select.POLLIN)
    if self.audio_fifo:
      self.audio_poll = select.poll()
      self.audio_poll.register(self.audio_fifo,select.POLLIN)
    print('ffmpeg pid:',self.pipe.pid)

#
# This is the queue that holds everything together
#
  def __init__(self, source, queueSize=2048):

    self.streams = []
    self.stream = source
    self.image = False
    if self.stream.endswith('.jpg') or self.stream.endswith('.png'):
      self.image = True
    self.name = 'input_%d_%d' % (os.getpid(),random.choice(range(1,1000)))

    self.display = False
    if os.environ.get('DISPLAY'):
      self.display = True

    self.caption_guide = {}
    self.filename = None
    self.state = None
    self.pipe = None
    #self.transcribe = None
    self.video_fifo = None
    self.audio_fifo = None
    self.caption_fifo = None
    self.age = 0
    self.filepos = 0
    self.clockbase = 0
    self.microclockbase = 0
    self.stopped = multiprocessing.Value('i',0)
    self.ff = False

    # disabled for now. 
    self.tas = None
    #self.tas = TranscribeAudioStream().load()

    # initialize the analyzer pipe
    #self.Q = queue.Queue(maxsize=queueSize)
    self.Q = multiprocessing.Queue(maxsize=queueSize)

  def __del__(self):
    if self.pipe:
      # not cleanly exiting the threads leads to all kinds of problems, including deadlocks
      outs, errs = self.pipe.communicate(timeout=15)
      self.pipe.poll()
      self.pipe.wait()
      self.pipe.terminate()
      self.pipe.kill()

    if self.video_fifo:
      os.close(self.video_fifo)
    if self.audio_fifo:
      os.close(self.audio_fifo)
    if self.caption_fifo:
      os.close(self.audio_fifo)

    # I need to figure out how to get it to reset the console too
    if self.video_fifo and os.path.exists('/tmp/%s_video' % self.name):
      os.unlink('/tmp/%s_video' % self.name)
    if self.audio_fifo and os.path.exists('/tmp/%s_audio' % self.name):
      os.unlink('/tmp/%s_audio' % self.name)
    if self.caption_fifo and os.path.exists('/tmp/%s_caption' % self.name):
      os.unlink('/tmp/%s_caption' % self.name)

  def load(self,width,height,fps,sample,scale=0.2):
    self.scale = scale
    self.width = width
    self.height = height
    self.fps = fps
    self.sample = sample
    self.t = multiprocessing.Process(target=self.update, args=())
    self.t.daemon = True
    self.t.start()
    return self

# this is the demuxer.  it puts the video frames into
#  a queue.  the captions go into neverending hash (never gets pruned)
#  audio goes into a neverending place in memory
  def update(self):
     print("process!",self.stopped.value)
     self.process()
     timeout = 0
     read_frame = 0

     start_talking = 0
     no_talking = 0

     data_buf = []
     last_buf = None

     play_audio = bytearray()
     raw_audio = bytes()
     raw_image = bytes()

     while self.stopped.value == 0:
       bstart = time.time() 

       # TODO: replace with qmax for both
       if self.Q.qsize() == 1023:
         print('running fast, sleeping')
       if self.Q.qsize() >= 1024:
         time.sleep(0.01)
         
       if self.audio_poll is None and self.video_poll is None and self.image is False:
         print('done')
         break

       if True:
         rstart = time.time()

# TODO: get this into utils/var
         data = new_frame(read_frame)
         data['fps'] = self.fps

###### captions
#  4096 was kind of chosen at random.  Its a nonblocking read
         if self.caption_fifo and self.caption_poll is not None:
           # doesn't really need to do this on every frame 
           events = self.caption_poll.poll(1)
           if len(events) > 0 and not events[0][1] & select.POLLIN:
             raw_subs = os.read(self.caption_fifo,4096)
             self.parse_caption(raw_subs)

###### audio
#  audio is read in, a little bit at a time
         if self.audio_fifo and self.audio_poll is not None and data.get('audio') is None:
           fail = 0
           bufsize = int((self.sample*2/self.fps) - len(raw_audio))
           events = self.audio_poll.poll(1)
           if len(events) > 0 and not events[0][1] & select.POLLIN:
             self.audio_poll.unregister(self.audio_fifo)
             self.audio_poll = None
             os.close(self.audio_fifo)
             self.audio_fifo = None
             print('warning audio error',events[0][1])
             continue

           while bufsize > 0 and len(events) > 0 and events[0][1] & select.POLLIN:
             tmp = os.read(self.audio_fifo,bufsize)
             events = self.audio_poll.poll(1)
             if tmp is not None:
               raw_audio += tmp
               bufsize = int((2 * self.sample / self.fps) - len(raw_audio))
               if bufsize < 0:
                 print('warning, negative buf',bufsize)
             else:
               fail += 1
               print("warning audio underrun0",len(raw_audio))
               time.sleep(0.001)
             if fail > 5:
               print("warning audio underrun1",len(raw_audio))
               break
           if raw_audio is not None and len(raw_audio) == int(self.sample*2/self.fps):
             data['audio'] = raw_audio
             raw_audio = bytes()

         if data is not None and data.get('audio') is not None:
             data['audio_np'] = np.fromstring(data['audio'],np.int16)
             data['audio_level'] = np.std(data['audio_np'])
             longest = 0
             last = 0
               
             for i,e in enumerate(data['audio_np']):
               if e == 0:
                 longest += 1
                 if longest > last:
                   last = longest
               else:
                 longest = 0

             data['silence'] = last
             data['speech_level'] = speech_calc(self,data['audio_np'])
             play_audio += data['audio']
             if data['speech_level'] > 0.8:
               #print("Talking!",data['speech_level'],read_frame)
               start_talking = read_frame
               no_talking = 0
             else:
               no_talking += 1
             if (no_talking == 10 and len(play_audio) > int(2*self.sample)):
               #print("Talking Break!", (read_frame - start_talking) / self.fps,len(play_audio))

               # BUG: This can sometimes take forever to run and crash.  There is some bad math in there somewhere
               #tmp = speech_calc(self,np.frombuffer(play_audio,np.int16))
               # this needs to be multithreaded
               #print("  Transcribing:",len(play_audio),read_frame,read_frame - start_talking)
               if self.tas and self.tas.transcribe:
                 self.tas.transcribe.stdin.write(play_audio)
               play_audio = bytearray()
               start_talking = 0

###### video
         if self.image:
           data['raw'] = cv2.resize(cv2.imread(self.stream),(self.width,self.height))
         elif self.video_fifo and self.video_poll is not None and data.get('raw') is None:
           fail = 0
           bufsize = self.width*self.height*3 - len(raw_image)
           events = self.video_poll.poll(1)
           if len(events) > 0 and not events[0][1] & select.POLLIN:
             self.video_poll.unregister(self.video_fifo)
             self.video_poll = None
             os.close(self.video_fifo)
             self.video_fifo = None
             print('warning video error',events[0][1])
             continue

           while bufsize > 0 and len(events) > 0 and events[0][1] & select.POLLIN:
             tmp = os.read(self.video_fifo,bufsize)
             events = self.video_poll.poll(1)
             if tmp is not None:
               raw_image += tmp
               bufsize = self.width*self.height*3 - len(raw_image)
             else:
               fail += 1
               print("warning video underrun0",len(raw_image))
               time.sleep(0.001)
             if fail > 5:
               print("warning video underrun1",len(raw_image))
               break
           if raw_image is not None and len(raw_image) == self.width*self.height*3:
             data['raw'] = np.fromstring(raw_image,dtype='uint8').reshape((self.height,self.width,3))
             raw_image = bytes()

         if data is not None and data.get('raw') is not None:
           # crop out letter and pillarbox
           #  don't do this if we get a null  - its a scene break
           #  TODO: don't do this unless the height and width is changing by a lot
           start = time.time()
           data['small'] = cv2.resize(data['raw'],(int(data['raw'].shape[:2][1] * self.scale),int(data['raw'].shape[:2][0] * self.scale)))
           non_empty_columns = np.int0(np.where(data['small'].max(axis=0) > 0)[0] / self.scale)
           if len(non_empty_columns) > 100 and min(non_empty_columns) > self.height * 0.05 and max(non_empty_columns) < self.height * 0.95:
             data['rframe'] = data['raw'][0:self.height,min(non_empty_columns):max(non_empty_columns)+1:]
             data['small'] = cv2.resize(data['rframe'],(int(data['rframe'].shape[:2][1] * self.scale),int(data['rframe'].shape[:2][0] * self.scale)))
           else:
             data['rframe'] = data['raw']

           data['height'], data['width'], data['channels'] = data['rframe'].shape
           data['scale'] = self.scale

           data['gray'] = cv2.cvtColor(data['small'],cv2.COLOR_BGR2GRAY)
           data['hsv'] = cv2.cvtColor(data['small'],cv2.COLOR_BGR2HSV)

           data['lum'] = np.sum(data['hsv']) / float(data['hsv'].shape[0] * data['hsv'].shape[1] * data['hsv'].shape[2])
           data['frame_mean'] = np.sum(data['small']) / float(data['small'].shape[0] * data['small'].shape[1] * data['small'].shape[2])
           data['hist'] = cv2.calcHist([data['small']], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
           data['hist'] = cv2.normalize(data['hist'],5).flatten()
           hist = data['hist']
           hist = hist.ravel()/hist.sum()
           logs = np.log2(hist+0.00001)
           data['contrast'] = -1 * (hist*logs).sum()


###### transcription
         if self.tas and self.tas.transcribe and self.tas.transcribe.stdout:
           frame_transcribe = self.tas.transcribe.stdout.read(4096)
           if frame_transcribe is not None:
             data['transcribe'] = frame_transcribe.decode('ascii')
             print("  Transcribe:",len(play_audio),frame_transcribe)


         # drop frames if necessary, only if URL 
         if self.audio_fifo and self.video_fifo:
           if data.get('audio') is not None and data.get('rframe') is not None:
             self.microclockbase += 1 / self.fps
             # instead of directly playing the video, buffer it a little
             # buffer this for 30 frames and send them all at the same time
             #  this way we can stitch the audio together 
             window = self.fps
             if read_frame % (2*window) == 0 and read_frame > 0:
               cast_audio = bytes()
               for buf in data_buf[0:2*window]:
                 cast_audio += buf['audio']
               data_buf[0]['play_audio'] = cast_audio

               # TODO: if no audio, pad with zero bytes
               # TODO: peek ahead, find blank frames, break candidates
               #  TODO: move last_buf to previous buf block

               # this is basically a NMS algorithm.  Find all the upcoming breaks
               #  and take the strongest one
               new_buf = []
               breaks = []
               data_buf.append(data)
               for buf in data_buf:
                 if last_buf is not None and buf['sbd'] == 0.0:
                   buf = shot.frame_to_contours(buf,last_buf)
                   buf = shot.frame_to_shots(buf,last_buf)
                 if buf['shot_detect'] == buf['frame']:
                   breaks.append((buf['frame'],buf['sbd']))
                 last_buf = buf
                 new_buf.append(buf)

               #   should be pretty easy
               real_break = new_buf[0]['shot_detect']
               if len(breaks) > 0:
                 breaks.sort(key=lambda x: x[1],reverse=True)
                 real_break = breaks[0][0]

               for buf in new_buf[0:2*window]:
                 buf['shot_detect'] = real_break
                 self.Q.put_nowait(buf)
               data_buf = new_buf[2*window:]
             else:
               data_buf.append(data)
             read_frame += 1
         elif self.video_fifo or self.image:
           if data.get('rframe') is not None:
             self.microclockbase += 1 / self.fps
             self.Q.put_nowait(data)
             read_frame += 1
      
         time.sleep(0.0001)

     if self.video_fifo:
       os.close(self.video_fifo)
     if self.audio_fifo:
       os.close(self.audio_fifo)

  def read(self):
    return self.Q.get()

  def stop(self):
    print('stop!')
    self.stopped.value = 1
    for i in range(self.Q.qsize()):
      self.Q.get()
    self.Q = None

#
# Turn the output from the lavfi filter into a dict with timestamps
# TODO: move this wherever it goes
#
  def parse_caption(self,line):
    clean = line.rstrip().decode('ascii', errors='ignore')
    cleanr = re.compile('<.*?>')
    clean = re.sub(cleanr,'',clean)
    cleanr = re.compile('\{.an\d\}')
    clean = re.sub(cleanr,' ',clean)
    last_caption = ''
    for line in clean.splitlines():
      mo = re.search("^(\d\d:\d\d:\d\d),\d+? \-\-\> (\d\d:\d\d:\d\d),\d+",line)
      if mo and mo.group():
        last_caption = datetime.datetime.strftime(datetime.datetime.strptime(mo.group(1),"%H:%M:%S") + datetime.timedelta(seconds=self.clockbase),'%H:%M:%S')
        self.caption_guide[last_caption] = {}
        # add this to the internal frame clock
        self.caption_guide[last_caption]['start'] = last_caption
        self.caption_guide[last_caption]['stop'] = datetime.datetime.strftime(datetime.datetime.strptime(mo.group(2),"%H:%M:%S") + datetime.timedelta(seconds=self.clockbase),'%H:%M:%S')
        self.caption_guide[last_caption]['caption'] = ''
      elif last_caption:
        if line.isdigit():
          self.caption_guide[last_caption]['scene'] = int(line) - 1
        else:
          self.caption_guide[last_caption]['caption'] += line + ' '

