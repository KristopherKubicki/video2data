#!/usr/bin/python3 
# coding: utf-8
#

#
# Placeholder for eventual hashing algorithm
#  Right now its lossless compression
#
def phash_bits(bitstring):
  if bitstring is not '0b':
    tmp = zlib.compress(bitstring.encode())
    #tmp = int(bitstring, 2).to_bytes((len(bitstring) + 7) // 8, 'big')
    # first 100 
    return base64.urlsafe_b64encode(tmp).decode('utf-8')[:-2][:100]

#
# General timing configurations for queue sizes. Simple time shifting
#
def smoothWait():
  waitkey = int(1000 / FPS)

  qsize = fvs.Q.qsize()
  if qsize > 320:
    waitkey -= 36
  elif qsize > 300:
    waitkey -= 32
  elif qsize > 280:
    waitkey -= 28
  elif qsize > 260:
    waitkey -= 24
  elif qsize > 240:
    waitkey -= 20
  elif qsize > 120:
    waitkey -= 14
  elif qsize > 80:
    waitkey -= 12
  elif qsize > 60:
    waitkey -= 8
  elif qsize > 45:
    waitkey -= 4
  elif qsize <= 10:
    waitkey += 60
  elif qsize < 5:
    print("sleepy time")
    time.sleep(1)
    waitkey += 120

  if waitkey < 1 or fvs.ff:
    waitkey = 1

  return waitkey
