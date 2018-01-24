#!/usr/bin/python3 
# coding: utf-8

# I had to manually patch this to make it work for python3
#  there are better ways to do it
# https://bugs.launchpad.net/python-v4l2/+bug/1664158
import v4l2
from fcntl import ioctl

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
    if self.device:
      self.device.close()

  def load(self,width,height):

    print("make sure you ran: modprobe v4l2loopback")
    self.device = open('/dev/video0', 'wb')
    capability = v4l2.v4l2_capability()
    fmt = v4l2.V4L2_PIX_FMT_YUYV
    format = v4l2.v4l2_format()
    format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
    format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_BGR24
    format.fmt.pix.width = width
    format.fmt.pix.height = height
    format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    format.fmt.pix.bytesperline = width * 3
    format.fmt.pix.sizeimage = width * height * 3
    ioctl(self.device, v4l2.VIDIOC_S_FMT, format)
    return self

  def cast(self,buf):
    self.device.write(buf)
