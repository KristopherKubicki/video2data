#!/usr/bin/python3 

load up the im2txt captioner.  This is used to describe shots
#  mine is trained against mscoco.  This will be a rapidly advancing area
sys.path.append('/home/kristopher/models/research/im2txt')
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
imcap_graph = tf.Graph()
with imcap_graph.as_default():
  model = inference_wrapper.InferenceWrapper()
  restore_fn = model.build_graph_from_config(configuration.ModelConfig(),'/home/kristopher/models/research/im2txt/im2txt/model/train/')
  imcap_graph.finalize()
  # TODO: this graph could benefit from being frozen.  Compression + speed enhancements
  vocab = vocabulary.Vocabulary('/home/kristopher/models/research/im2txt/im2txt/data/word_counts.txt')
  imcap_sess = tf.Session(graph=imcap_graph)
  restore_fn(imcap_sess)
  generator = caption_generator.CaptionGenerator(model, vocab,4,17,1.5)
  #image_enc = cv2.imencode('.jpg', image_np)[1].tostring()
  #captions = generator.beam_search(imcap_sess, image_enc)
  #print("Captions for test image:")
  #for i, caption in enumerate(captions):
  #  sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
  #  sentence = " ".join(sentence)
  #  print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

