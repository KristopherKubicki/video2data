#!/usr/bin/python3 
# coding: utf-8

#
# Another model that can detect tons of objects. This is currently
# the biggest public model that I know about,  Open Images.  It has 546 objects
#  It takes 0.7s per detection.  12s to load
# 
print("loading open images, this takes a while")
# http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2017_11_08.tar.gz
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2017_11_08'
# the low proposals set is supposed to be twice as fast as the normal one.  i have found that not be the case
# and they run about the same.  
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2017_11_08'
PATH_TO_CKPT = '/home/kristopher/tf_files/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
oi_sess = None
oi_graph = tf.Graph()
with oi_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='oi')

# this is the coco set now
NUM_CLASSES = 546
olabel_map = label_map_util.load_labelmap('/home/kristopher/tf_files/models/' + MODEL_NAME + '/oid_bbox_trainable_label_map.pbtxt')
ocategories = label_map_util.convert_label_map_to_categories(olabel_map, max_num_classes=NUM_CLASSES, use_display_name=True)
ocategory_index = label_map_util.create_category_index(ocategories)

oi_image_tensor = oi_graph.get_tensor_by_name('oi/image_tensor:0')
oi_detection_boxes = oi_graph.get_tensor_by_name('oi/detection_boxes:0')
oi_detection_scores = oi_graph.get_tensor_by_name('oi/detection_scores:0')
oi_detection_classes = oi_graph.get_tensor_by_name('oi/detection_classes:0')
oi_num_detections = oi_graph.get_tensor_by_name('oi/num_detections:0')
oi_sess = tf.Session(graph=oi_graph)

(oboxes, oscores, oclasses, onum) = oi_sess.run(
 [oi_detection_boxes, oi_detection_scores, oi_detection_classes, oi_num_detections],
  feed_dict={oi_image_tensor: image_np_expanded})
