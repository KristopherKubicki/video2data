#!/usr/bin/python3 
# 
# Scene text detection
#

#EAST super fast text detection
#  I had to symlink nets from the EAST/nets -- not sure why.  I will need to follow up on this 
# TODO: freeze model
sys.path.append('/home/kristopher/EAST')
import model as east
import loadeast
input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
f_score, f_geometry = east.model(input_images, is_training=False)
ocr_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
saver = tf.train.Saver(variable_averages.variables_to_restore())
saver.restore(ocr_sess, '/home/kristopher/EAST/east_icdar2015_resnet_v1_50_rbox/model.ckpt-49491')

