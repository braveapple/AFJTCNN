
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

from Dataloader import *
from pre_model import CNN_Encoder
slim=tf.contrib.slim



class Config():
    batch_size = 128

    learning_rate=0.001

    gt_file='trainCASIA.txt'



loader = Dataloader(Config.batch_size,Config.gt_file)


class AFJT():
    
    def __init__(self, config,input=None):
        
        # Hyperparameter
        self.batch_size = batch_size = config.batch_size
        #self.categorical_size = config.categorical_size
        #self.embed_size = config.embed_size
        
        self.image_height = 112
        self.image_width = 96
        self.label1num=12575
        #self.label2num=5000
        self.ratio=0.008
        
        self.learning_rate = config.learning_rate
        

        # Placeholder
        
        self.images = tf.placeholder(tf.float32,
                                     shape=[batch_size, self.image_height, self.image_width, 3],
                                     name='input_images')
        
        #self.text = tf.placeholder(tf.int32,
        #                             shape=[batch_size, self.seq_len], 
        #                             name='text')
        #self.mask = tf.placeholder(tf.float32,
        #                          shape=[batch_size, self.seq_len],
        #                          name='mask')
        self.label1=tf.placeholder(tf.int64,shape=[batch_size])
        #self.label2=tf.placeholder(tf.int32,shape=[batch_size])
        self.training = tf.placeholder(tf.bool, name='training')
        
        
        # Encoder and decoder
        logits,features = CNN_Encoder(self.label1num,self.images, self.training)
        
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
            with tf.name_scope('softmax_loss'):
                self.softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label1, logits=logits))
            with tf.name_scope('total_loss'):
                self.loss = self.softmax_loss + self.ratio * self.center_loss
    
        with tf.name_scope('acc'):
            print self.label1,tf.arg_max(logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), self.label1), tf.float32))
    
        with tf.name_scope('loss/'):
            tf.summary.scalar('CenterLoss', self.center_loss)
            tf.summary.scalar('SoftmaxLoss', self.softmax_loss)
            tf.summary.scalar('TotalLoss', self.loss)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(centers_update_op)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)



        

    

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
            model.saver.restore(sess=sess, save_path='Model3/AFJT-' + str(start))

        import time

        start_time = time.time()

        average_loss = 0.0
        average_loss1 = 0.0
        average_loss2 = 0.0
        average_acc = 0.0
        
        logw=open('loglfw.log','w')
        for i in range(start, 1000000):

            x_batch, y_batch= loader.get_batch(i % loader.batch_num)
            # x_batch, y_batch, mask, seq_len=get_m_train_batch(Config.batch_size)

            feed_dict = {model.images: x_batch, model.label1: y_batch, model.training: True}

            loss,loss1,loss2, acc, _ = sess.run([model.loss, model.softmax_loss,model.center_loss,model.accuracy, model.train_op], feed_dict=feed_dict)

            

            average_loss = average_loss + (loss / 100)
            average_loss1 = average_loss1 + (loss1 / 100)
            average_loss2 = average_loss2 + (loss2 / 100)
            average_acc = average_acc + (acc / 100)
            

            if (i) % 100 == 0:
                print(i, ' loss ', average_loss,average_loss1,average_loss2, ' acc ', average_acc * 100.0)
                logw.write(str(i )+' loss '+ str(average_loss)+' '+str(average_loss1)+' '+str(average_loss2)+ ' acc '+str(average_acc * 100.0)+'\n')
                end_time = time.time()
                print('Second Per Batch: %2f' % ((end_time - start_time) / 100.0))
                start_time = end_time

                average_loss = 0.0
                average_loss1 = 0.0
                average_loss2 = 0.0
                average_acc = 0.0
                

                # if i % 10 == 0 :
                #     j = (i // 2)%(12800// Config.batch_size)
                #     x_batch, y_batch, mask, seq_len = get_m_test_batch(Config.batch_size, j)
                #     feed_dict = { model.images:x_batch, model.text:y_batch,model.training:False, model.mask:mask}
                #     loss, acc, x=sess.run([model.loss, model.accuracy, model.gen_x], feed_dict=feed_dict)

                # err = np.mean(get_edit_distance_lst(y_batch,x))

                # agent.append(Test_Loss, i, float(loss))
                # agent.append(Test_Acc, i, float(acc))
                # agent.append(Test_ED, i, float(err))

                # if i % 100 == 0:
                #         print('')
                #         print('VTrue: ',show_text(y_batch))
                #         print('VPred: ', show_text(x))
                #         print(i,'Vloss ', loss,' acc ', acc)#, ' err ', err)

            if (i + 1) % 1000 == 0 and i != start:
                model.saver.save(sess, 'Model3/AFJT', global_step=i + 1)
                print('save model')


                # In[ ]:



