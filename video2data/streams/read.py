#!/usr/bin/python3 
# coding: utf-8
 
import os
import random
import threading
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
    return
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
      self.video_fifo.close()
    if self.audio_fifo is not None:
      self.audio_fifo.close()
    if self.caption_fifo is not None:
      self.caption_fifo.close()

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
    # 
    if self.stream[:4] != 'rtsp':
      # probably should do this better
      self.stream = str.replace(self.stream,':','\\\:')
      self.stream = str.replace(self.stream,'@','\\\@')
 
      # we should probably always stick to 're' for most video
      video_command = [ 'ffmpeg','-nostdin','-re','-hide_banner','-loglevel','panic','-hwaccel','vdpau','-y','-f', 'lavfi','-i','movie=%s:s=0\\\:v+1\\\:a[out0+subcc][out1]' % self.stream,'-map','0:v','-vf','scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2' % (self.width,self.height,self.width,self.height),'-pix_fmt','bgr24','-r','%f' % self.fps,'-s','%dx%d' % (self.width,self.height),'-vcodec','rawvideo','-f','rawvideo','/tmp/%s_video' % self.name, '-map','0:a','-acodec','pcm_s16le','-r','%f' % self.fps,'-ab','16k','-ar','%d' % self.sample,'-ac','1','-f','wav','/tmp/%s_audio' % self.name, '-map','0:s','-f','srt','/tmp/%s_caption' % self.name  ] 
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
      self.caption_fifo = open('/tmp/%s_caption' % self.name,'rb',4096)
      flags = fcntl(self.caption_fifo, F_GETFL)
      fcntl(self.caption_fifo, F_SETFL, flags | os.O_NONBLOCK)
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
      self.video_fifo = open('/tmp/%s_video' % self.name,'rb',self.width*self.height*3*10)
    else:
      print("unrecognized input")
      return
 
    if self.video_fifo:
      self.video_poll = select.poll()
      self.video_poll.register(self.video_fifo,select.POLLIN)
    if self.audio_fifo:
      self.audio_poll = select.poll()
      self.audio_poll.register(self.audio_fifo,select.POLLIN)

    time.sleep(5)
    print('pid:',self.pipe.pid)

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
    self.stopped = False
    self.ff = False

    # disabled for now. 
    self.tas = None
    #self.tas = TranscribeAudioStream().load()

    # initialize the analyzer pipe
    self.Q = queue.Queue(maxsize=queueSize)

  def __del__(self):
    # I need to figure out how to get it to reset the console too
    if os.path.exists('/tmp/%s_video' % self.name):
      os.unlink('/tmp/%s_video' % self.name)
    if os.path.exists('/tmp/%s_audio' % self.name):
      os.unlink('/tmp/%s_audio' % self.name)
    if os.path.exists('/tmp/%s_caption' % self.name):
      os.unlink('/tmp/%s_caption' % self.name)
    if self.video_fifo:
      self.video_fifo.flush()
      self.video_fifo.close()
    if self.audio_fifo:
      self.audio_fifo.flush()
      self.audio_fifo.close()
    if self.caption_fifo:
      self.caption_fifo.flush()
      self.caption_fifo.close()
    if self.pipe:
      # not cleanly exiting the threads leads to all kinds of problems, including deadlocks
      outs, errs = self.pipe.communicate(timeout=15)
      self.pipe.poll()
      self.pipe.wait()
      self.pipe.terminate()
      self.pipe.kill()

# Thread here
  def load(self,width,height,fps,sample,scale=0.2):
    self.scale = scale
    self.width = width
    self.height = height
    self.fps = fps
    self.sample = sample
    self.t = threading.Thread(target=self.update, args=())
    self.t.daemon = True
    self.t.start()
    return self

# this is the demuxer.  it puts the video frames into
#  a queue.  the captions go into neverending hash (never gets pruned)
#  audio goes into a neverending place in memory
  def update(self):
     print("process!")
     self.process()
     timeout = 0
     read_frame = 0

     start_talking = 0
     no_talking = 0

     play_audio = bytearray()
     raw_audio = bytes()
     raw_image = bytes()

     while self.stopped is False:
       if self.Q.qsize() >= 1024:
         time.sleep(0.1)

       if self.Q.qsize() < 1024:
         rstart = time.time()
         data = {}
         data['speech_level'] = 0.0
         data['fps'] = self.fps

###### captions
#  4096 was kind of chosen at random.  Its a nonblocking read
         if self.caption_fifo:
           raw_subs = self.caption_fifo.read(4096)
           if raw_subs is not None:
             self.parse_caption(raw_subs)

###### audio
#  audio is read in, a little bit at a time
         if self.audio_fifo and self.audio_poll is not None and len(self.audio_poll.poll(1)) > 0:
           raw_audio = os.read(self.audio_fifo,int(2*self.sample / self.fps))
           #if raw_audio is None:
           #  print('warning audio frame empty')
           if raw_audio is not None:
             data['audio'] = raw_audio
             raw_audio = bytes()
             #data['audio_np'] = np.frombuffer(data['audio'],np.int16)
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
         elif self.video_fifo and self.video_poll is not None and len(self.video_poll.poll(1)) > 0:
           fail = 0
           bufsize = self.width*self.height*3 - len(raw_image)

           while bufsize > 0 and self.video_poll is not None and len(self.video_poll.poll(1)) > 0:
             tmp = os.read(self.video_fifo,bufsize)
             if tmp is not None:
               raw_image += tmp
               bufsize = self.width*self.height*3 - len(raw_image)
               if bufsize < 0:
                 print('warning, negative buf',bufsize)
             else:
               fail += 1
               time.sleep(0.01)
             if fail > 5:
               #print("warning video underrun1",len(raw_image))
               break
           if raw_image is not None and len(raw_image) == self.width*self.height*3:
             data['raw'] = np.fromstring(raw_image,dtype='uint8').reshape((self.height,self.width,3))

         if data is not None and data.get('raw') is not None:
           raw_image = bytes()
           cv2.imwrite('/tmp/tmp.bmp',data['raw'])
           data['bw'] = cv2.cvtColor(data['raw'], cv2.COLOR_BGR2GRAY)
           non_empty_columns = np.where(data['bw'].max(axis=0) > 0)[0]
           non_empty_rows = np.where(data['bw'].max(axis=1) > 0)[0]
           # crop out letter and pillarbox
           #  don't do this if we get a null  - its a scene break
           #  TODO: don't do this unless the height and width is changing by a lot
           if len(non_empty_rows) > 200 and len(non_empty_columns) > 200 and min(non_empty_columns) > 0:
             #data['rframe'] = data['raw'][min(non_empty_rows):max(non_empty_rows)+1,min(non_empty_columns):max(non_empty_columns)+1:]
             data['rframe'] = data['raw'][0:self.height,min(non_empty_columns):max(non_empty_columns)+1:]
           else:
             data['rframe'] = data['raw']

           data['height'], data['width'], data['channels'] = data['rframe'].shape
           data['frame'] = data['rframe']
           data['scale'] = self.scale
           data['frame_mean'] = np.sum(data['rframe']) / float(data['rframe'].shape[0] * data['rframe'].shape[1] * data['rframe'].shape[2])
           data['hist'] = cv2.calcHist([data['rframe']], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
           data['hist'] = cv2.normalize(data['hist'],5).flatten()
           hist = data['hist']
           hist = hist.ravel()/hist.sum()
           logs = np.log2(hist+0.00001)
           data['contrast'] = -1 * (hist*logs).sum()

           data['small'] = cv2.resize(data['rframe'],(int(data['rframe'].shape[:2][1] * self.scale),int(data['rframe'].shape[:2][0] * self.scale)))
           data['gray'] = cv2.cvtColor(data['small'],cv2.COLOR_BGR2GRAY)
         else:
           time.sleep(1/self.fps)
           #raw_image = bytes()
           #print("warning video underrun2",len(raw_image))
           continue


###### transcription
         if self.tas and self.tas.transcribe and self.tas.transcribe.stdout:
           #print('reading3')
           frame_transcribe = self.tas.transcribe.stdout.read(4096)
           #print('reading4')
           if frame_transcribe is not None:
             data['transcribe'] = frame_transcribe.decode('ascii')
             print("  Transcribe:",len(play_audio),frame_transcribe)
##
         self.microclockbase += 1 / self.fps
         read_frame += 1
         # drop frames if necessary, only if URL 
         #if self.stream[:4] == 'rtsp' or self.stream[:4] == 'http') and self.Q.qsize() < 1000:
         self.Q.put(data)
         time.sleep(0.01)

     os.close(self.video_fifo)
     os.close(self.audio_fifo)

  def read(self):
    return self.Q.get()

  def stop(self):
    print('stop!')
    self.stopped = True

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

