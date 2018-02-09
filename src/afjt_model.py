
# coding: utf-8

# In[4]:

import tensorflow as tf
import numpy as np
# import pickle

from tensorflow import layers

# from PIL import Image
#from my_util import *



slim=tf.contrib.slim

def CNN_Encoder(label1num,label2num, images, training,reuse=False):
        
    layers.conv2d = tf.contrib.framework.add_arg_scope(layers.conv2d)
    cnn_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
    cnn_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
#   cnn_initializer = tf.orthogonal_initializer(gain=0.01)
                
    with tf.contrib.framework.arg_scope([layers.conv2d],reuse=reuse,
                                   kernel_regularizer=cnn_regularizer, 
                                   bias_regularizer=cnn_regularizer,
                                   kernel_initializer=cnn_initializer):
            net = layers.conv2d(images, 32, [3,3], name='conv1_0',padding='valid')
            net = layers.batch_normalization(net, name='conv1_bn0', training=training)
            net = tf.nn.relu(net)
            net = layers.conv2d(net, 64, [3,3], name='conv1_1',padding='valid')
            net = layers.batch_normalization(net, name='conv1_bn1', training=training)
            net = tf.nn.relu(net, name='conv1_relu')
            #[bs,108,92,64]
            net = layers.max_pooling2d(net, [2,2],[2,2], name='pool_1')
            print net.shape
            #[bs,54,46,64]
            net2 = layers.conv2d(net, 64, [3,3], name='conv2_0',padding='same')
            net2 = layers.batch_normalization(net2, name='conv2_bn0', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 64, [3,3], name='conv2_1',padding='same')
            net2 = layers.batch_normalization(net2, name='conv2_bn1', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = layers.conv2d(net3, 128, [3,3], name='conv2_2',padding='valid')
            net = layers.batch_normalization(net, name='conv2_bn2', training=training)
            net = tf.nn.relu(net)
            #[bs 52 44 128]
            net = layers.max_pooling2d(net,[2,2], [2,2], name='pool_2')
            #[bs 26 22 128]
            net2 = layers.conv2d(net, 128, [3,3], name='conv3_0',padding='same')
            net2 = layers.batch_normalization(net2, name='conv3_bn0', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 128, [3,3], name='conv3_1',padding='same')
            net2 = layers.batch_normalization(net2, name='conv3_bn1', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3
            net2 = layers.conv2d(net, 128, [3,3], name='conv3_2',padding='same')
            net2 = layers.batch_normalization(net2, name='conv3_bn2', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 128, [3,3], name='conv3_3',padding='same')
            net2 = layers.batch_normalization(net2, name='conv3_bn3', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = layers.conv2d(net3, 256, [3,3], name='conv3_4',padding='valid')
            net = layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)
            #[bs 24 20 256]
            net = layers.max_pooling2d(net,[2,2], [2,2], name='pool_3')
            print net.shape
            #[bs 12 10 256]
            net2 = layers.conv2d(net, 256, [3,3], name='conv4_0',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn0', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 256, [3,3], name='conv4_1',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn1', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3
            net2 = layers.conv2d(net, 256, [3,3], name='conv4_2',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn2', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 256, [3,3], name='conv4_3',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn3', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3
            net2 = layers.conv2d(net, 256, [3,3], name='conv4_4',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn4', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 256, [3,3], name='conv4_5',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn5', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3
            net2 = layers.conv2d(net, 256, [3,3], name='conv4_6',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn6', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 256, [3,3], name='conv4_7',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn7', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3
            net2 = layers.conv2d(net, 256, [3,3], name='conv4_8',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn8', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 256, [3,3], name='conv4_9',padding='same')
            net2 = layers.batch_normalization(net2, name='conv4_bn9', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = layers.conv2d(net3, 512, [3,3], name='conv4_10',padding='valid')
            net = layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)
            #[bs 10 8 512]
            net = layers.max_pooling2d(net,[2,2], [2,2], name='pool_4')
            print net.shape
            #[bs 5 4 512]
            net2 = layers.conv2d(net, 512, [3,3], name='conv5_0',padding='same')
            net2 = layers.batch_normalization(net2, name='conv5_bn0', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 512, [3,3], name='conv5_1',padding='same')
            net2 = layers.batch_normalization(net2, name='conv5_bn1', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3
            net2 = layers.conv2d(net, 512, [3,3], name='conv5_2',padding='same')
            net2 = layers.batch_normalization(net2, name='conv5_bn2', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 512, [3,3], name='conv5_3',padding='same')
            net2 = layers.batch_normalization(net2, name='conv5_bn3', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3
            net2 = layers.conv2d(net, 512, [3,3], name='conv5_4',padding='same')
            net2 = layers.batch_normalization(net2, name='conv5_bn4', training=training)
            net2 = tf.nn.relu(net2)
            net2 = layers.conv2d(net2, 512, [3,3], name='conv5_5',padding='same')
            net2 = layers.batch_normalization(net2, name='conv5_bn5', training=training)
            net2 = tf.nn.relu(net2)
            net3 = tf.add(net,net2)
            net = net3

            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=512, activation_fn=None, scope='fc5')
            feature=slim.fully_connected(net, num_outputs=512, activation_fn=None, scope='fc6-1')
            logits1 = slim.fully_connected(feature, num_outputs=label1num, activation_fn=None, scope='sflabel1')
            aux=slim.fully_connected(net, num_outputs=512, activation_fn=None, scope='fc6-2')
            logits2 = slim.fully_connected(aux, num_outputs=label2num, activation_fn=None, scope='sflabel2')
                
                
        
    return net,logits1,logits2,feature