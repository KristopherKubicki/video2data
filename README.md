# video2data

<b>video2data</b> or <b>v2d</b> is a general purpose tool for describing images, video and audio.  The project aggregates multiple state-of-the-art neural networks in order to describe the contents of the video stream for machines or humans. 

<img src='https://user-images.githubusercontent.com/478212/35297058-7ecca3e8-0043-11e8-884c-9466b08701b4.png'>

The project started as a way to detect and recognize individuals in a security camera feed.  With recent breakthroughs in computer vision, machine learning and hardware acceleration, creating an end-to-end *understanding* of what's going in a video or image seems within reach.  v2d aims to integrate many of these technologies quickly for unconstrained environments. 

The project relies heavily on Tensorflow and open-sourced neural network models.

This repository is highly unstable and going through a prototype-to-mvp refactor.  

# Features

v2d converts a multimedia stream and output events to stdout.  Over time, the project plans to implement MQTT and caption-file outputs.  

## Realtime Reading

v2d mimics how the human brain works by first recognizing blocks of text in a scene.  It reads each block from largest to smallest, left to right, as fast as it can.  For more constrained environments, v2d will still spot the text but only translate if its been on the scene for a while. 

<img src='https://user-images.githubusercontent.com/478212/35297123-b0e94412-0043-11e8-8dd6-3e5ab7399c71.png'>

## Scene Understanding

Using several breakthroughs in machine learning, v2d recognizes hundreds of common objects in images.  It then uses novel mechanisms for summarizing those objects for easy archiving or event notifications.  v2d also keeps track of when a scene 'breaks' or switches to another viewing angle. 

<img src='https://user-images.githubusercontent.com/478212/35297627-3062abd8-0045-11e8-8700-9c98c4caf639.png'>

## Face, Person, Celebrity Recognition

Detecting people and faces is an accurate and reliable way to archive video and imagery.  v2d assigns every face it sees with a unique ID.  Every time v2d sees a face, it will check to see how similar that new face is to all the other faces it has already scene.  

<img src='https://user-images.githubusercontent.com/478212/35298918-3e0646ba-0049-11e8-8220-923dcaf64c82.png'>

## High-Speed Tracking

Humans do not re-evaluate what an object is after they are reasonably sure of what they see.  v2d emulates this behavior by detecting an object, and then accurately tracking it.  It classifies certain objects as high-speed candidates and follows them accordingly. 

<img src='https://user-images.githubusercontent.com/478212/35297270-1d5bc502-0044-11e8-9077-9986d23a3f65.png'>

## Custom Object Recognition

Some of the newest advances in object recognition allow us to recognize just about any object or *concept*. Looking for candy in a video segment?  Maybe just 'happy scenes'?  All of these are possible with some of the newest models. 

# Requirements

The v2d codebase relies heavily on GPU-optimized libraries.  Please review requirements.txt for the full list.  In addition, a powerful computer is recommended.  

# Installation

The project aims to develop a docker and/or kubernetes image.  Currently it has very heavy installation requirements.  You should only consider running this application on an NVIDIA 1070 or higher, and a 2016 i7 or better processor. 

Due to the way the program pipes FIFO traffic between processes, it is recommended to only use Linux.  My development machine is an Ubuntu 17.10 release.

# Usage

v2d has many applications ranging from security and home automation to advertising and analytics. 

```
$ v2d.py --input garden.jpg
```
Runs v2d against an image, taking extra time to enhance the image, recognizing objects and summarizing the scene as a caption.  v2d will also take additional time to do deep text recognition, converting an written words to text.

<img src='https://user-images.githubusercontent.com/478212/35298114-bf1fbda6-0046-11e8-99c1-a5dd2ad12f12.png'>

```
$ v2d.py --input videofile.mpg
```

v2d will run against a video file.  Since v2d uses ffmpeg, almost any video format is natively supported.  This is useful for creating descriptions of video for archiving and easy recall. In addition, v2d is fingerprinting all shots and scenes it detects by default

<img src='https://user-images.githubusercontent.com/478212/35298409-bb758734-0047-11e8-9f8b-8a1fd7205cc7.png'>

```
$ v2d.py --input rtsp://example.com/yourstream
```

v2d is well suited to capture and annotate webstream for security and automation.  


<img src='https://user-images.githubusercontent.com/478212/35298663-7f320a30-0048-11e8-8667-03f621dd6da1.png'>

```
$ v2d.py --input http://example.com/livestream.webm --output /dev/video0
```

Loads a v4l device and writes the video frames to the device.  This is a cheap and easy way to record what's going on inside v2d without transcoding inside the program.

Much of the v2d development was done on live streaming HDHomerun MPEG-TS endpoints.   It works very well on live TV as well as more static sources. 

# Roadmap

## Deep Audio Recognition
There are breakthroughs every day in this space.  The newest models detect common sounds very well: music, glass breaking, gunshots, humans.  Speech to Text is coming in a big way too.
## Very Advanced Scene Recognition
Some of the newest models can describe any image in exacting, deep detail.  We will see these soon.

# Contributing

I am actively developing v2d and plan to incorporate many advanced features.  If you would like to participate in its development, please feel free to issue a PR or reach out!

# Thanks

Special thanks to all the people who inspired me to build this application.  
<ul>
  <li><a href='https://github.com/tensorflow/tensorflow'>Tensorflow</a></li>
  <li><a href='https://github.com/tensorflow/models/tree/master/research/object_detection'>Tensorflow Object Detection</a></li>
  <li><a href='https://github.com/tensorflow/models/tree/master/research/im2txt'>Show and Tell</a></li>
  <li><a href='https://pjreddie.com/darknet/yolo/'>YOLO</a></li>
  <li><a href='https://github.com/tensorflow/models/tree/master/research/attention_ocr'>Attention OCR</a></a>
  <li><a href='https://github.com/davisking/dlib'>dlib</a></a></li>
  <li><a href='https://www.pyimagesearch.com/'>Adrian Rosebrock</a></li>
 </ul>

