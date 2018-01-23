#!/usr/bin/python3 
# coding: utf-8

# now load up the fastest model. Average inference is about 0.03s at 1080p
print("loading mobilenet")
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#  Mobilenet is a lot faster!
#MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
PATH_TO_CKPT = '/home/kristopher/tf_files/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
coco_graph = tf.Graph()
with coco_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='coco')
NUM_CLASSES = 90
clabel_map = label_map_util.load_labelmap('/home/kristopher/tf_files/models/' + MODEL_NAME + '/mscoco_label_map.pbtxt')
ccategories = label_map_util.convert_label_map_to_categories(clabel_map, max_num_classes=NUM_CLASSES, use_display_name=True)
ccategory_index = label_map_util.create_category_index(ccategories)

c_image_tensor = coco_graph.get_tensor_by_name('coco/image_tensor:0')
c_detection_boxes = coco_graph.get_tensor_by_name('coco/detection_boxes:0')
c_detection_scores = coco_graph.get_tensor_by_name('coco/detection_scores:0')
c_detection_classes = coco_graph.get_tensor_by_name('coco/detection_classes:0')
c_num_detections = coco_graph.get_tensor_by_name('coco/num_detections:0')

coco_sess = tf.Session(graph=coco_graph)

start = time.time()
(cboxes, cscores, cclasses, cnum) = coco_sess.run(
  [c_detection_boxes, c_detection_scores, c_detection_classes, c_num_detections],
  feed_dict={c_image_tensor: image_np_expanded})
print("init coco_done",time.time() - start)
