#!/usr/bin/python3 
# coding: utf-8

#
# A car / pedestrian favored model, good for exterior cameras
#  It takes 0.04s for 1080p
#  
print("loading KITTI (car, people)")
MODEL_NAME = 'faster_rcnn_resnet101_kitti_2017_11_08'
# http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2017_11_08.tar.gz
PATH_TO_CKPT = '/home/kristopher/tf_files/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
kitti_graph = tf.Graph()
with kitti_graph.as_default():
  kitti_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='kitti')

# this is the coco set now
NUM_CLASSES = 2
klabel_map = label_map_util.load_labelmap('/home/kristopher/tf_files/models/' + MODEL_NAME + '/kitti_label_map.pbtxt')
kcategories = label_map_util.convert_label_map_to_categories(klabel_map, max_num_classes=NUM_CLASSES, use_display_name=True)
kcategory_index = label_map_util.create_category_index(kcategories)

kitti_image_tensor = kitti_graph.get_tensor_by_name('kitti/image_tensor:0')
kitti_detection_boxes = kitti_graph.get_tensor_by_name('kitti/detection_boxes:0')
kitti_detection_scores = kitti_graph.get_tensor_by_name('kitti/detection_scores:0')
kitti_detection_classes = kitti_graph.get_tensor_by_name('kitti/detection_classes:0')
kitti_num_detections = kitti_graph.get_tensor_by_name('kitti/num_detections:0')

kitti_sess = tf.Session(graph=kitti_graph)
(kboxes, kscores, kclasses, knum) = kitti_sess.run(
  [kitti_detection_boxes, kitti_detection_scores, kitti_detection_classes, kitti_num_detections],
  feed_dict={kitti_image_tensor: image_np_expanded})
print("init kitti_done",time.time() - start)


