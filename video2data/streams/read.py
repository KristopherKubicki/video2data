#!/usr/bin/python3 
# coding: utf-8
 
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
      video_command = [ 'ffmpeg','-nostdin','-re','-hide_banner','-loglevel','info','-hwaccel','vdpau','-y','-f', 'lavfi','-i','movie=%s:s=0\\\:v+1\\\:a[out0+subcc][out1]' % self.stream,'-map','0:v','-vf','scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2' % (WIDTH,HEIGHT,WIDTH,HEIGHT),'-pix_fmt','bgr24','-r','%f' % FPS,'-s','%dx%d' % (WIDTH,HEIGHT),'-vcodec','rawvideo','-f','rawvideo','/tmp/%s_video' % self.name, '-map','0:a','-acodec','pcm_s16le','-r','%f' % FPS,'-ab','16k','-ar','%d' % SAMPLE,'-ac','1','-f','wav','/tmp/%s_audio' % self.name, '-map','0:s','-f','srt','/tmp/%s_caption' % self.name  ] 
      self.pipe = subprocess.Popen(video_command)

      print('Step 2 initializing video /tmp/%s_video' % self.name)
      if not os.path.exists('/tmp/%s_video' % self.name):
        return
      #self.video_fifo = open('/tmp/%s_video' % self.name,'rb',WIDTH*HEIGHT*3*10)
      self.video_fifo = os.open('/tmp/%s_video' % self.name,os.O_RDONLY | os.O_NONBLOCK)
      #fcntl(self.video_fifo,1031,6220800)
      fcntl(self.video_fifo,1031,1048576)
      #flags = fcntl(self.video_fifo, F_GETFL)
      #fcntl(self.video_fifo, F_SETFL, flags | O_NONBLOCK)

      print('Step 3 initializing video /tmp/%s_video' % self.name)
      #self.audio_fifo = open('/tmp/%s_audio' % self.name,'rb',SAMPLE*10)
      self.audio_fifo = os.open('/tmp/%s_audio' % self.name,os.O_RDONLY | os.O_NONBLOCK,SAMPLE*10)
      fcntl(self.audio_fifo,1031,1048576)
      # buffer size has been an issue before too
      #fcntl(self.audio_fifo,1031,6220800)
      #fcntl(self.audio_fifo,1031,1048576)
      flags = fcntl(self.audio_fifo, F_GETFL)
      fcntl(self.audio_fifo, F_SETFL, flags | O_NONBLOCK)

      self.caption_fifo = open('/tmp/%s_caption' % self.name,'rb',4096)
      flags = fcntl(self.caption_fifo, F_GETFL)
      fcntl(self.caption_fifo, F_SETFL, flags | O_NONBLOCK)
    #
    # big assumption rtsp stream is h264 of some kind and probably a security camera
    #  it does not handle audio, as the audio not actually in the stream.  Presumably we could mux it here if it exists
    elif self.stream[:4] == 'rtsp':
      # we don't need the lavfi command because there are no subs.  also, the movie= container for rtsp doesn't let me pass rtsp_transport, which will result in
      #  dropped packets if I do not do
      # also, vdpau is not working for some reason (yuv issues)
      # 
      video_command = [ 'ffmpeg','-nostdin','-re','-hide_banner','-loglevel','panic','-y','-r','%f' % FPS,'-rtsp_transport','tcp','-i',self.stream,'-map','0:v','-vf','scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2' % (WIDTH,HEIGHT,WIDTH,HEIGHT),'-pix_fmt','bgr24','-r','%f' % FPS,'-vcodec','rawvideo','-f','rawvideo','/tmp/%s_video' % self.name ]

      self.pipe = subprocess.Popen(video_command)
      print('fifo: /tmp/%s_video' %self.name)
      self.video_fifo = open('/tmp/%s_video' % self.name,'rb',WIDTH*HEIGHT*3*10)
    else:
      print("unrecognized input")
      return
 
    self.audio_poll = select.poll()
    self.video_poll = select.poll()
    self.audio_poll.register(self.audio_fifo,select.POLLIN)
    self.video_poll.register(self.video_fifo,select.POLLIN)

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
    self.stopped = multiprocessing.Value('i',False)
    self.ff = False

    # disabled for now. 
    self.tas = None
    #self.tas = TranscribeAudioStream().load()
    self.ocr = OCRStream().load()

    # initialize the analyzer pipe
    self.Q = multiprocessing.Queue(maxsize=queueSize)

  def __del__(self):
    # I need to figure out how to get it to reset the console too
    if os.path.exists('/tmp/%s_video' % self.name):
      os.unlink('/tmp/%s_video' % self.name)
    if os.path.exists('/tmp/%s_audio' % self.name):
      os.unlink('/tmp/%s_audio' % self.name)
    if os.path.exists('/tmp/%s_caption' % self.name):
      os.unlink('/tmp/%s_caption' % self.name)
    if self.t:
      self.t.terminate()
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
  def load(self):
    #self.t = threading.Thread(target=self.update, args=())
    self.t = multiprocessing.Process(target=self.update, args=())
    #self.t.daemon = True
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


     print('reading1...',self.stopped.value,self.t.pid,self.Q.qsize(),self.t.is_alive())
     while self.stopped.value == 0 and self.t.is_alive():
       print('reading2...',self.stopped.value,self.t.pid,self.Q.qsize(),self.t.is_alive())
       if self.Q.qsize() >= 1024:
         time.sleep(0.1)
         print('timing out...')

       if self.Q.qsize() < 1024:
         rstart = time.time()
         data = {}
         data['speech_level'] = 0.0

###### captions
#  4096 was kind of chosen at random.  Its a nonblocking read
         if self.caption_fifo:
           raw_subs = self.caption_fifo.read(4096)
           if raw_subs is not None:
             self.parse_caption(raw_subs)

###### audio
#  audio is read in, a little bit at a time
         #if self.audio_fifo and self.pipe.poll() is None and self.audio_poll.poll() is None:
         if self.audio_fifo and self.audio_poll is not None and len(self.audio_poll.poll(1)) > 0:
           print('reading audio1',int(2*SAMPLE/FPS),self.audio_poll.poll(1))
           #raw_audio = self.audio_fifo.read(int(2*SAMPLE / FPS))

           raw_audio = os.read(self.audio_fifo,int(2*SAMPLE / FPS))
           print('reading audio2')
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
             if (no_talking == 10 and len(play_audio) > int(2*SAMPLE)):
               #print("Talking Break!", (read_frame - start_talking) / FPS,len(play_audio))

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
           data['raw'] = cv2.resize(cv2.imread(self.stream),(WIDTH,HEIGHT))
         #elif self.video_fifo and self.pipe.poll() is None and self.video_poll.poll() is None:
         elif self.video_fifo and self.video_poll is not None and len(self.video_poll.poll(1)) > 0:
           fail = 0
           bufsize = WIDTH*HEIGHT*3 - len(raw_image)

           #print('reading video1')
           while bufsize > 0:
             if data.get('audio'):
               print('reading video1a',bufsize,'/tmp/%s_video' % self.name,len(data.get('audio')),self.video_poll.poll(1))
             else:
               print('reading video1a',bufsize,'/tmp/%s_video' % self.name,0,self.video_poll.poll(1))

             #tmp = self.video_fifo.read(bufsize)
             tmp = os.read(self.video_fifo,bufsize)
             print('reading video1b')
             if tmp is not None:
               raw_image += tmp
               bufsize = WIDTH*HEIGHT*3 - len(raw_image)
               if bufsize < 0:
                 print('warning, negative buf',bufsize)
             else:
               fail += 1
               time.sleep(0.01)
             if fail > 5:
               #print("warning video underrun1",len(raw_image))
               break
           if raw_image is not None and len(raw_image) == WIDTH*HEIGHT*3:
             data['raw'] = np.fromstring(raw_image,dtype='uint8').reshape((HEIGHT,WIDTH,3))

         print('reading2a...',self.stopped.value,self.t.pid,self.Q.qsize(),self.t.is_alive(),self.pipe.poll())
         if data is not None and data.get('raw') is not None:
           print('reading2a1')
           raw_image = bytes()
           print('reading2a1a1',len(data['raw']),read_frame)
           cv2.imwrite('/tmp/tmp.bmp',data['raw'])
           cv2.imshow('buf',data['raw'])
           cv2.waitKey(100)
           data['bw'] = cv2.cvtColor(data['raw'], cv2.COLOR_BGR2GRAY)
           print('reading2a1a2')
           non_empty_columns = np.where(data['bw'].max(axis=0) > 0)[0]
           print('reading2a1a3')
           non_empty_rows = np.where(data['bw'].max(axis=1) > 0)[0]
           print('reading2a1a4')
           # crop out letter and pillarbox
           #  don't do this if we get a null  - its a scene break
           #  TODO: don't do this unless the height and width is changing by a lot
           print('reading2a1a')
           if len(non_empty_rows) > 200 and len(non_empty_columns) > 200 and min(non_empty_columns) > 0:
             #data['rframe'] = data['raw'][min(non_empty_rows):max(non_empty_rows)+1,min(non_empty_columns):max(non_empty_columns)+1:]
             data['rframe'] = data['raw'][0:HEIGHT,min(non_empty_columns):max(non_empty_columns)+1:]
           else:
             data['rframe'] = data['raw']

           data['height'], data['width'], data['channels'] = data['rframe'].shape
           data['frame'] = data['rframe']
           print('reading2a1b')
           data['frame_mean'] = np.sum(data['rframe']) / float(data['rframe'].shape[0] * data['rframe'].shape[1] * data['rframe'].shape[2])
           data['hist'] = cv2.calcHist([data['rframe']], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
           data['hist'] = cv2.normalize(data['hist'],5).flatten()
           print('reading2a1c')
           hist = data['hist']
           hist = hist.ravel()/hist.sum()
           logs = np.log2(hist+0.00001)
           data['contrast'] = -1 * (hist*logs).sum()

           print('reading2a1d')

           data['small'] = cv2.resize(data['rframe'],(int(data['rframe'].shape[:2][1] * mser_scaler),int(data['rframe'].shape[:2][0] * mser_scaler)))
           data['gray'] = cv2.cvtColor(data['small'],cv2.COLOR_BGR2GRAY)
           print('reading2a1e')
         else:
           print('reading2a1')
           time.sleep(1/FPS)
           #raw_image = bytes()
           print("warning video underrun2",len(raw_image))
           continue


###### transcription
         if self.tas and self.tas.transcribe and self.tas.transcribe.stdout:
           print('reading3')
           frame_transcribe = self.tas.transcribe.stdout.read(4096)
           print('reading4')
           if frame_transcribe is not None:
             data['transcribe'] = frame_transcribe.decode('ascii')
             print("  Transcribe:",len(play_audio),frame_transcribe)
##
         self.microclockbase += 1 / FPS
         read_frame += 1
         # drop frames if necessary, only if URL 
         #if self.stream[:4] == 'rtsp' or self.stream[:4] == 'http') and self.Q.qsize() < 1000:
         print('done',read_frame)
         self.Q.put(data)
         time.sleep(0.01)

     os.close(self.video_fifo)
     os.close(self.audio_fifo)

  def read(self):
    return self.Q.get()

  def stop(self):
    print('stop!')
    self.stopped.value = 1

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

