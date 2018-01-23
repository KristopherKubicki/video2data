#!/usr/bin/python3 
# coding: utf-8
#
# converts Audio to Text
#   eventually use a neural network instead of pocketsphinx
#
class TranscribeAudioStream:

  def __init__(self):
    self.transcribe = None

  def __del__(self):
    if self.transcribe is not None:
      self.transcribe.kill()

  def load(self):
     # this should be threadable...
     self.transcribe = subprocess.Popen(['pocketsphinx_continuous','-infile','/dev/stdin'], stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
     flags = fcntl(self.transcribe.stdout, F_GETFL)
     fcntl(self.transcribe.stdout, F_SETFL, flags | O_NONBLOCK)
     return self
