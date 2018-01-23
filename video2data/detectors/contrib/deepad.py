#!/usr/bin/python3 

# load up the SSD-derived NN I trained on the 16 brand logos
#   this is used for commercial detection.  Average inference is about 0.03s on 1080p
MODEL_NAME = 'da1_12_20_2017'
PATH_TO_CKPT = '/home/kristopher/tf_files/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
logo_graph = tf.Graph()
with logo_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
NUM_CLASSES = 16
label_map = label_map_util.load_labelmap('/home/kristopher/tf_files/tmp/deepad.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# create all the object detection return value structures
da_image_tensor = logo_graph.get_tensor_by_name('image_tensor:0')
da_detection_boxes = logo_graph.get_tensor_by_name('detection_boxes:0')
da_detection_scores = logo_graph.get_tensor_by_name('detection_scores:0')
da_detection_classes = logo_graph.get_tensor_by_name('detection_classes:0')
da_num_detections = logo_graph.get_tensor_by_name('num_detections:0')

logo_sess = tf.Session(graph=logo_graph)

(boxes, scores, classes, num) = logo_sess.run(
   [da_detection_boxes, da_detection_scores, da_detection_classes, da_num_detections],
    feed_dict={da_image_tensor: image_np_expanded})

