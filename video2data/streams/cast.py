#!/usr/bin/python3 
# coding: utf-8

# 
# Initialize casting buffer, so we can proxy out whaver feed came in
#
# Writes to a fifo, which another subprocess picks up with ffmpeg
#  transcodes rawvideo (and eventually audio, caption chanels)
#  into webm or some other codec. 
#
class CastVideoStream:

  def __init__(self):
    self.device = None
 
  def __del__(self):
    if self.pipe is not None:
      self.pipe.terminate()
    if self.video_fifo is not None:
      self.video_fifo.flush()
      self.video_fifo.close()
    if self.audio_fifo is not None:
      self.audio_fifo.flush()
      self.audio_fifo.close()
    if os.path.exists('/tmp/%s_video' % self.name):
      os.unlink('/tmp/%s_video' % self.name)
    if os.path.exists('/tmp/%s_audio' % self.name):
      os.unlink('/tmp/%s_audio' % self.name)

  def load(self):

    print("make sure you ran: modprobe v4l2loopback")
    self.device = open('/dev/video0', 'wb')
    capability = v4l2_capability()
    fmt = V4L2_PIX_FMT_YUYV
    format = v4l2_format()
    format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24
    format.fmt.pix.width = WIDTH
    format.fmt.pix.height = HEIGHT
    format.fmt.pix.field = V4L2_FIELD_NONE
    format.fmt.pix.bytesperline = WIDTH * 3
    format.fmt.pix.sizeimage = WIDTH * HEIGHT * 3
    ioctl(self.device, VIDIOC_S_FMT, format)
    return self

  def cast(self,buf):
    self.device.write(buf)
