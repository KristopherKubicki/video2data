#!/usr/bin/python3
# coding: utf-8

import sys

#load up the im2txt captioner.  This is used to describe shots
#  mine is trained against mscoco.  This will be a rapidly advancing area
#
# consider just importing this whole thing 
sys.path.append('/home/kristopher/models/research/im2txt')
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

import tensorflow as tf
import cv2
import math

class nnIm2txt:

  def load(self):
    print("loading image captioning")
    self.imcap_graph = tf.Graph()
    with self.imcap_graph.as_default():
      model = inference_wrapper.InferenceWrapper()
      restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                           'models/im2txt/')
      self.imcap_graph.finalize()
      # TODO: this graph could benefit from being frozen.  Compression + speed enhancements
      self.vocab = vocabulary.Vocabulary('models/im2txt/word_counts.txt')
      self.sess = tf.Session(graph=self.imcap_graph)
      restore_fn(self.sess)
      self.generator = caption_generator.CaptionGenerator(model, self.vocab,4,17,1.5)

    return self
 
  def run(self,image_np,include_labels=[],exclude_labels=[]):

    shot_summary = ''
    image_enc = cv2.imencode('.jpg', image_np)[1].tostring()
    # NOTE: I have implemented modifications to the beam search. 
    #  this needs to be brought into the codebase
    captions = self.generator.beam_search(self.sess, image_enc,include_labels,exclude_labels)
    captions.sort(key=lambda x: x.score)
    for i, caption in enumerate(captions):
      sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      shot_summary = ' '.join(sentence).replace(' .','.').replace(' \'s','\'s')
      break
    return shot_summary

  def test(self,image_np):
    self.run(image_np)


