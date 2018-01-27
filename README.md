# video2data

<b>video2data</b> or <b>v2d</b> is a general purpose tool for describing images, video and audio.  v2d aggregates multiple state-of-the-art neural networks to classify contents of the video stream for machines or humans. 

<a href='https://www.youtube.com/watch?v=Hq8-2D2iGok&feature=youtu.be'><img src='https://user-images.githubusercontent.com/478212/35297058-7ecca3e8-0043-11e8-884c-9466b08701b4.png'>Click to Watch a Neural Network Watch TV</a>

The project started as a way to detect and recognize individuals in a security camera feed.  Recent breakthroughs in computing bring <b>*understanding*</b>.  v2d aims to integrate many bleeding-edge technologies quickly for unconstrained environments. 

This codebase relies heavily on Tensorflow and open-sourced neural-network models.  The repository is highly unstable and going through a prototype-to-mvp refactor.  

# Features

v2d converts a multimedia stream and output events to stdout.  Over time, the project plans to implement MQTT and caption-file outputs.  

## Realtime Reading

v2d mimics how the human brain works by first recognizing blocks of text in a scene.  It reads each block from largest to smallest, left to right, as fast as it can.  For more constrained environments, v2d will still spot the text but only translate if it can keep up with the signal. 

<img src='https://user-images.githubusercontent.com/478212/35303963-76733174-0059-11e8-9759-2f2644ddb71d.png'>

## Scene Understanding

v2d recognizes hundreds of common objects in images.  It then uses novel mechanisms for summarizing those objects for easy archiving or event notifications.  v2d also keeps track of when a scene 'breaks' or switches to another viewing angle. 

<img src='https://user-images.githubusercontent.com/478212/35297627-3062abd8-0045-11e8-8700-9c98c4caf639.png'>

## Face, Person, Celebrity Recognition

Detecting people and faces is an accurate and reliable way to archive video and imagery.  v2d assigns every face it sees with a unique ID.  It will check to see how similar that new face is to all the other faces it has already scene.  

<img src='https://user-images.githubusercontent.com/478212/35298918-3e0646ba-0049-11e8-8220-923dcaf64c82.png'>

## High-Speed Tracking

Humans do not re-evaluate what an object is after they are reasonably sure.  v2d emulates this behavior by tracking objects after its first noticed them.  It classifies certain objects as high-speed candidates and follows them accordingly. 

<img src='https://user-images.githubusercontent.com/478212/35297270-1d5bc502-0044-11e8-9077-9986d23a3f65.png'>

## Custom Object Recognition

Some of the newest advances in object recognition allow us to recognize just about any object or *concept*. Looking for candy in a video segment?  Maybe just 'happy scenes'?  All of these are possible with some of the newest models. 

# Requirements

* python3
* tesseract 4.00.00alpha
    leptonica-1.74.4
      libjpeg 8d (libjpeg-turbo 1.5.2) : libpng 1.6.34 : libtiff 4.0.8 : zlib 1.2.11
* ffmpeg version 3.3.4-2


The v2d codebase relies heavily on GPU-optimized libraries.  Please review requirements.txt for the full list.  In addition, a powerful computer is recommended.  

# Installation

The project aims to develop a docker and/or kubernetes image.  Currently it has very heavy installation requirements.  You should only consider running this application on an NVIDIA 1070 or higher, and a 2016 i7 or better processor. 

Due to the way the program pipes FIFO traffic between processes, it is recommended to only use Linux.  My development machine is Ubuntu 17.10.

1. Clone repo
2. $ cd video2data/models/
2. Download <a href='https://drive.google.com/uc?export=download&id=1rdh6dNliIIOOdrX_zY-7ruam8Kzc13yZ'>pre-trained models</a>
3. $ tar zxvf models.tgz .
4. $ cd ..
5. $ ./v2d.py

# Usage

v2d has many applications ranging from security and home automation to advertising and analytics. 

```
$ v2d.py --input garden.jpg
```
Runs v2d against an image, taking extra time to enhance the image, recognizing objects and summarizing the scene as a caption.  v2d will also take additional time to do deep text recognition, converting any written words to text.

<img src='https://user-images.githubusercontent.com/478212/35298114-bf1fbda6-0046-11e8-99c1-a5dd2ad12f12.png'>

```
$ v2d.py --input videofile.mpg
```

Runs v2d against a video file.  Since v2d uses ffmpeg, almost any video format is natively supported.  This is useful for creating descriptions of video for archiving and easy recall. In addition, v2d is fingerprinting all shots and scenes it detects by default.

<img src='https://user-images.githubusercontent.com/478212/35298409-bb758734-0047-11e8-9f8b-8a1fd7205cc7.png'>

```
$ v2d.py --input rtsp://example.com/yourstream
```

Runs v2d against a remote stream.  The program is well suited to capture and annotate a webstream for security and automation.  


<img src='https://user-images.githubusercontent.com/478212/35298663-7f320a30-0048-11e8-8667-03f621dd6da1.png'>

```
$ v2d.py --input http://example.com/livestream.webm --output /dev/video0
```

Runs v2d against a remote MPEG stream, and then loads a v4l device and writes the video frames to the device.  This is a cheap and easy way to record what's going on.

Much of the v2d development was done on live streaming HDHomerun MPEG-TS endpoints.   It works very well on live TV as well as more static sources. 

# Roadmap

## Initial Release
Moving from a 2300 line prototype to a fully open source model
## Deep Audio Recognition
There are breakthroughs every day in this space.  The newest models detect common sounds very well: music, glass breaking, gunshots, humans.  Speech to Text is coming in a big way too.
## Natural Scene Recognition
Some of the newest models can describe any image in exacting, deep detail.  We will see these soon.

# High Level

<img src='https://user-images.githubusercontent.com/478212/35318197-e3fce618-009f-11e8-8f96-9ee33241f8be.png'>

# Contributing

I am actively developing v2d and plan to incorporate many new features.  If you would like to participate in its development, please feel free to issue a PR or reach out!

# Thanks

Special thanks to all the people who inspired and helped me build this application.  
<ul>
  <li><a href='https://github.com/tensorflow/tensorflow'>Tensorflow</a></li>
  <li><a href='https://github.com/tensorflow/models/tree/master/research/object_detection'>Tensorflow Object Detection</a></li>
  <li><a href='https://github.com/tensorflow/models/tree/master/research/im2txt'>Show and Tell</a></li>
  <li><a href='https://pjreddie.com/darknet/yolo/'>YOLO</a></li>
  <li><a href='https://github.com/tensorflow/models/tree/master/research/attention_ocr'>Attention OCR</a></a>
  <li><a href='https://github.com/davisking/dlib'>dlib</a></a></li>
  <li><a href='https://www.pyimagesearch.com/'>Adrian Rosebrock</a></li>
  <li><a href='http://blueprintalpha.com/'>Blueprint Alpha</a></li>
  <li><a href='https://soundcloud.com/hoodinternet'>The Hood Internet</a></li>
  <li>#hassbro</li>
  <li><a href='https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A'>Siraj Raval</a></li>
 </ul>

