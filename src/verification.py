
# coding: utf-8

# In[4]:

import tensorflow as tf
import numpy as np
# import pickle

from tensorflow import layers


import os
import cv2
import sys
import os.path as osp
from afjt_model import CNN_Encoder


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

#from Dataloader2 import *
slim=tf.contrib.slim


class Config():
    batch_size = 2

    learning_rate=0.0000

    gt_folder='/home/d201/cropcacdvs/'


class Attention_OCR():
    
    def __init__(self, config,input=None):
        
        # Hyperparameter
        self.batch_size = batch_size = config.batch_size
        
        self.image_height = 112
        self.image_width = 96
        self.label1num=2000
        self.label2num=5046
        self.ratio=0.008
        
        self.learning_rate = config.learning_rate
        

        # Placeholder
        
        self.images = tf.placeholder(tf.float32,
                                     shape=[batch_size, self.image_height, self.image_width, 3],
                                     name='input_images')
        
        self.label1=tf.constant(value=[0,0],dtype=tf.int64)
        self.label2=tf.constant(value=[0,0],dtype=tf.int64)
        

        self.training = tf.placeholder(tf.bool, name='training')
        
        
        # Encoder and decoder
        _,l1,l2,self.features = CNN_Encoder(self.label1num,self.label2num,self.images, self.training)
        
        def prelu(_x, name=None):   
            if name is None:     
                name = "alpha"
            alpha = tf.get_variable(name,shape=_x.get_shape(),initializer=tf.constant_initializer(0.0),dtype=_x.dtype)    
            return tf.maximum(_alpha*_x, _x)

        def get_center_loss(features, labels, alpha, num_classes):

            len_features = features.get_shape()[1]
            centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                initializer=tf.constant_initializer(0), trainable=False)
            labels = tf.reshape(labels, [-1])
            centers_batch = tf.gather(centers, labels)
            loss = tf.nn.l2_loss(features - centers_batch)
            diff = centers_batch - features
   
            unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
    
            diff = diff / tf.cast((1 + appear_times), tf.float32)
            diff = alpha * diff
    
            centers_update_op = tf.scatter_sub(centers, labels, diff)
    
            return loss, centers, centers_update_op
        
        with tf.name_scope('loss'):
            with tf.name_scope('center_loss'):
                self.center_loss, centers, centers_update_op = get_center_loss(self.features, self.label1, 1, self.label1num)
            with tf.name_scope('softmax_loss1'):
                self.softmax_loss1= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label1, logits=l1))
            with tf.name_scope('softmax_loss2'):
                self.softmax_loss2= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label2, logits=l2))
            with tf.name_scope('total_loss'):
                self.loss = self.softmax_loss1+self.softmax_loss2 + self.ratio * self.center_loss
    
        with tf.name_scope('acc1'):
            print self.label1,tf.arg_max(l1, 1)
            self.accuracy1 = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(l1, 1), self.label1), tf.float32))
        with tf.name_scope('acc2'):
            print self.label2,tf.arg_max(l2, 1)
            self.accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(l2, 1), self.label2), tf.float32))
    
        with tf.name_scope('loss/'):
            tf.summary.scalar('CenterLoss', self.center_loss)
            tf.summary.scalar('SoftmaxLoss1', self.softmax_loss1)
            tf.summary.scalar('SoftmaxLoss2', self.softmax_loss2)
            tf.summary.scalar('TotalLoss', self.loss)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(centers_update_op)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        pre_var_list=tf.global_variables()
        var_list=[var for var in pre_var_list if (not 'fc6-2' in var.name) and (not 'sflabel' in var.name) and (not 'center' in var.name)]
        #print 'Abandoned:',[var for var in pre_var_list if 'fc6' in var.name]

        self.saver = tf.train.Saver(var_list, max_to_keep=None)



        

def testEER():
    gto=open('logcacdvs207.log','r')
    lines=gto.readlines()
    dis=[]
    for line in lines:
        dis.append(float(line.split(' ')[1]))
    #print(len(dis),dis[0])
    sot=sorted(dis)
    fenge=sot[1999]
    label=[]
    for distance in dis:
        if distance>fenge:
            label.append(0)
        else:
            label.append(1)
    #print label
    count=0
    for i in xrange(0,10):
        for j in xrange(i*200,(i+1)*200):
            if i in [0,2,4,6,8]:
                if label[j]==0:
                    count+=1
            if i in [1,3,5,7,9]:
                if label[j]==1:
                    count+=1
    #print count
    return count




    
    

if __name__ == '__main__':
    # In[57]:

    tf.reset_default_graph()
    model = Attention_OCR(Config)





    Sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    Sess_config.gpu_options.allow_growth = True
    with tf.Session(config=Sess_config) as sess:

        tf.global_variables_initializer().run()
       
        #start = 0
        for start in range(160000,160001,10000):
            if start != 0:
                model.saver.restore(sess=sess, save_path='Model3/AFJT-' + str(start))

            import time

            start_time = time.time()
        
            logw=open('logcacdvs207.log','w')
            for i in range(0, 4000):
                batch_img=[]
                temp_img = cv2.imread(Config.gt_folder+'%04d_0.jpg'%i)
                temp_img = cv2.resize(temp_img, (96,112),
                                  interpolation=cv2.INTER_CUBIC)
                temp_img = (temp_img -127.5)/128
                batch_img.append(temp_img)
                temp_img = cv2.imread(Config.gt_folder+'%04d_1.jpg'%i)
                temp_img = cv2.resize(temp_img, (96,112),
                                  interpolation=cv2.INTER_CUBIC)
                temp_img = (temp_img -127.5)/128
                batch_img.append(temp_img)
                batch_img = np.array(batch_img)

            
            # x_batch, y_batch, mask, seq_len=get_m_train_batch(Config.batch_size)

                feed_dict = {model.images: batch_img, model.training: False}

                feature = sess.run([model.features], feed_dict=feed_dict)
                feature=np.squeeze(np.asarray(feature))
                #print feature.shape,type(feature),
                featuredistance=np.sum(np.square(feature[0]-feature[1]))
                #print featuredistance
                logw.write(str(i)+' '+str(featuredistance)+'\n')
            logw.close()
            eer=testEER()/4000.0
            print "ckpt:", start,'eer:',eer
