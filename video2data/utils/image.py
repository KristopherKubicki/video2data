#!/usr/bin/python3 
# coding: utf-8
#

# helper image for scene text detection
def resize_image(im, max_side_len=2400):
  h,w = im.shape[:2]
  resize_w = w
  resize_h = h
  if max(resize_h, resize_w) > max_side_len:
    ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
  else:
    ratio = 1.
  resize_h = int(resize_h * ratio)
  resize_w = int(resize_w * ratio)
  resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
  resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
  im = cv2.resize(im, (int(resize_w), int(resize_h)))
  ratio_h = resize_h / float(h)
  ratio_w = resize_w / float(w)
  return im, (ratio_h, ratio_w)


def zero_runs(a):
  iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
  absdiff = np.abs(np.diff(iszero))
  return np.where(absdiff == 1)[0].reshape(-1, 2)

def chi2_distance(histA, histB, eps = 1e-10):
  # compute the chi-squared distance
  d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
    for (a, b) in zip(histA, histB)])
  return d

def orderHull(pts):
  rect = np.zeros((4, 2), dtype = "float32")
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  return rect

def hullHorizontalImage(image, hull):
# format from neural net
  rect = orderHull(hull)
  (tl, tr, br, bl) = np.int0(rect)
  
  # if these 2 are close, don't skew it.  It's horizontal
  # 5% seems about right
  if (tr[1] - bl[1]) / image.shape[0] < 0.05:
    return image[tr[1]:bl[1],bl[0]:tr[0]:]

  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  return warped

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
  data_freq = np.fft.fftfreq(len(raw_audio),1.0/SAMPLE)
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

  # sometimes I get errors here
  return sum_energy / sum(energy_freq.values())

def rectInRect(rect1,rect2):
  return rect2[0] < rect1[0] < rect1[2] < rect2[2] and rect2[1] < rect1[1] < rect1[3] < rect2[3]

def pointInRect(point,rect):
  return rect[0] < point[0] < rect[2] and rect[1] < point[1] < rect[3]

def cntCenter(cnt):
  M = cv2.moments(cnt)
  if M["m00"] > 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY) 
  return (0,0)

def rectCenter(bbox):
  return (int((bbox[0] + bbox[2]) / 2),int((bbox[1] + bbox[3]) / 2))

def scaleRect(bbox,width,height,label=''):
  if bbox is None:
    return []
  if label:
    return (int(bbox[0] * width),int(bbox[1] * height),int(bbox[2] * width),int(bbox[3] * height),label)
  return (int(bbox[0] * width),int(bbox[1] * height),int(bbox[2] * width),int(bbox[3] * height))

def shiftRect(bbox):
  if bbox is None:
    return []
  return (bbox[1],bbox[0],bbox[3],bbox[2])

def normRect(bbox,width,height):
  if bbox is None:
    return []
  return (float(bbox[0] / width),float(bbox[1] / height),float(bbox[2] / width),float(bbox[3] / height))

def setLabel(im, label, y):
  fontface = cv2.FONT_HERSHEY_SIMPLEX
  scale = 0.6
  thickness = 1

  text_size = cv2.getTextSize(label, fontface, scale, thickness)[0]
  cv2.rectangle(im, (10, y - text_size[1] - 5), (10 + text_size[0], y + 5), (0,0,0), cv2.FILLED)
  cv2.putText(im, label ,(10, y),fontface, scale, (0, 255, 0), thickness)

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
