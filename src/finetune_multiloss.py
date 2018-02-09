
# coding: utf-8

# In[4]:

import tensorflow as tf
import numpy as np
# import pickle

from tensorflow import layers

# from PIL import Image
#from my_util import *

import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from Dataloader2 import *
slim=tf.contrib.slim
from afjt_model import CNN_Encoder



class Config():
    batch_size = 64

    learning_rate=0.0001
    gt_file='trainCACD.txt'



loader = Dataloader(Config.batch_size,Config.gt_file)


class AFJT():
    
    def __init__(self, config,input=None):
        
        # Hyperparameter
        self.batch_size = batch_size = config.batch_size
        #self.categorical_size = config.categorical_size
        #self.embed_size = config.embed_size
        
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
        

        self.label1=tf.placeholder(tf.int64,shape=[batch_size])
        self.label2=tf.placeholder(tf.int64,shape=[batch_size])
        self.training = tf.placeholder(tf.bool, name='training')
        
        
        # Encoder and decoder
        _,l1,l2,features = CNN_Encoder(self.label1num,self.label2num,self.images, self.training)
        
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
                self.center_loss, centers, centers_update_op = get_center_loss(features, self.label1, 1, self.label1num)
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
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(centers_update_op)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        pre_var_list=tf.global_variables()
        var_list=[var for var in pre_var_list if (not 'fc6-2' in var.name) and (not 'sflabel' in var.name) and (not 'center' in var.name)]
        self.saver2=tf.train.Saver(var_list, max_to_keep=None)
    
    

if __name__ == '__main__':
    # In[57]:

    tf.reset_default_graph()
    model = AFJT(Config)





    Sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    Sess_config.gpu_options.allow_growth = True
    with tf.Session(config=Sess_config) as sess:

        tf.global_variables_initializer().run()

        start = 0
        if start != 0:
            model.saver.restore(sess=sess, save_path='Modelcacdplus/AFJT-' + str(start))
        else:
            model.saver2.restore(sess=sess,save_path='Model3/AFJT-200000')

        import time

        start_time = time.time()

        average_loss = 0.0
        average_loss1 = 0.0
        average_loss2 = 0.0
        average_loss3 = 0.0
        average_acc1 = 0.0
        average_acc2 = 0.0
        
        logw=open('logcacd2.log','w')
        for i in range(start, 1000000):

            x_batch, y1_batch,y2_batch= loader.get_batch(i % loader.batch_num)

            feed_dict = {model.images: x_batch, model.label1: y1_batch, model.label2: y2_batch,model.training: True}

            loss,loss1,loss2,loss3, acc1,acc2, _ = sess.run([model.loss, model.softmax_loss1,model.softmax_loss2,model.center_loss,model.accuracy1,model.accuracy2, model.train_op], feed_dict=feed_dict)

            

            average_loss = average_loss + (loss / 100)
            average_loss1 = average_loss1 + (loss1 / 100)
            average_loss2 = average_loss2 + (loss2 / 100)
            average_loss3 = average_loss3 + (loss3 / 100)
            average_acc1 = average_acc1 + (acc1 / 100)
            average_acc12 = average_acc2 + (acc2 / 100)
            

            if (i) % 100 == 0:
                print(i, ' loss ', average_loss,average_loss1,average_loss2, average_loss3,' acc1 ', average_acc1 * 100.0,' acc2 ', average_acc2 * 100.0)
                logw.write(str(i )+' loss '+ str(average_loss)+' '+str(average_loss1)+' '+str(average_loss2)+' '+str(average_loss3)+ ' acc1 '+str(average_acc1 * 100.0)+' acc2 '+str(average_acc2 * 100.0)+'\n')
                end_time = time.time()
                print('Second Per Batch: %2f' % ((end_time - start_time) / 100.0))
                start_time = end_time

                average_loss = 0.0
                average_loss1 = 0.0
                average_loss2 = 0.0
                average_loss3 = 0.0
                average_acc1 = 0.0
                average_acc2 = 0.0
                


            if (i + 1) % 1000 == 0 and i != start:
                model.saver.save(sess, 'Model2/AFJT', global_step=i + 1)
                print('save model')


                # In[ ]:



