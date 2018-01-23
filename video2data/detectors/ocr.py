#!/usr/bin/python3 


# OCR class
class OCRStream:

  def load(self):
    #self.t = threading.Thread(target=self.update, args=())
    #multiprocessing.set_start_method('spawn')
    self.t = multiprocessing.Process(target=self.update, args=())
    self.t.daemon = True
    self.t.start()
    return self

  def update(self):

    while True:
      if self.input.qsize() == 0:
        time.sleep(0.01)
        continue

      # have to attach scene ID to each of these to make sure its needed
      text_id, image,text,highest_score = self.input.get()
      h,w,_ = image.shape
      cv2.imwrite('/tmp/ocr.bmp',image)
      ret,image = cv2.imencode(".bmp",image)
      #tesseract 4.00.00alpha
      # leptonica-1.74.4
      #   libjpeg 8d (libjpeg-turbo 1.5.2) : libpng 1.6.34 : libtiff 4.0.8 : zlib 1.2.11
      sp = subprocess.Popen(['tesseract','stdin','stdout','--oem','3','--psm','13','-l','eng','/home/kristopher/tf_files/scripts/tess.config'], stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
      #flags = fcntl(sp.stdout, F_GETFL)
      #fcntl(sp.stdout, F_SETFL, flags | O_NONBLOCK)
      tmp_data,err = sp.communicate(image.tostring())
      #tmp_data = ''
      sp.terminate()
      for line in tmp_data.decode('utf-8').split('\n'):
        cols = line.split('\t')
      #  #if len(cols) > 10 and cols[10].isdigit() and int(cols[10]) > highest_score and int(cols[8]) > w * 0.8:
        if len(cols) > 10 and cols[10].isdigit() and int(cols[10]) > int(highest_score):
      #     print('\t\tcols',cols[10],cols[11],text,highest_score)
           highest_score = int(cols[10])
           text = cols[11]
      
      self.output.put([text_id,text,highest_score])
      time.sleep(0.01)

  def __init__(self):
    self.input = multiprocessing.Queue(maxsize=1024)
    self.output = multiprocessing.Queue(maxsize=1024)

  def __del__(self):
    if self.t:
      self.t.terminate()

#
# alternative implementation to consider
#

# attention ocr
#ocr_graph = tf.Graph()
# TODO: this graph could benefit from being frozen
#model.ckpt-399731
#endpoints = model.create_base(images_placeholder, labels_one_hot=None)
#with ocr_graph.as_default():
#  model = inference_wrapper.InferenceWrapper()
#  restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
#          '/home/kristopher/models/research/attention_ocr/python/')
#ocr_graph.finalize()
#restore_fn(ocr_sess)
#ocr_image_tensor = ocr_graph.get_tensor_by_name('image_tensor:0')
#ocr_predictions = ocr_graph.get_tensor_by_name('image_tensor:0')

