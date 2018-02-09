
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
import fa
slim=tf.contrib.slim
from afjt_model import CNN_Encoder


# In[5]:

#show_text(np.array([1,2,3,4,5,6]).reshape(2,3))



# In[6]:

# img,text,mask,seq_len=get_m_train_batch(64)
# show_text(text)


# In[7]:

# Load character dict
# char_dict = {}
# char_dict = pickle.load(open('char_dict.pkl', 'rb'))
# index2char = char_dict['index2char']
# char2index = char_dict['char2index']
# index2char = [0,1,2,3,4,5,6,7,8,9,' ','STA','END']
# char2index = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9, ' ':10,'STA':11, 'END':12}
#index2char = ['0','1','2','3','4','5','6','7','8','9',
#				'STA','END', 'PAD']
#char2index = {'0':0,'1':1,'2':2,'3':3,'4':4,
#				'5':5,'6':6,'7':7,'8':8,'9':9,
#				'STA':10, 'END':11, 'PAD': 12}
class Config():
    batch_size = 64
    learning_rate=0.0001
    gt_file='trainCACD.txt'



loader = Dataloader(Config.batch_size,Config.gt_file)
loader_all=Dataloader_all(Config.batch_size,Config.gt_file)


class AFJT():
    
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
        self.label1=tf.placeholder(tf.int64,shape=[batch_size])
        self.label2=tf.placeholder(tf.int64,shape=[batch_size])
        self.training = tf.placeholder(tf.bool, name='training')
        
        
        # Encoder and decoder
        self.fc5o,l1,l2,features = CNN_Encoder(self.label1num,self.label2num,self.images, self.training)
        
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
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        
        #Notice:


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(centers_update_op)
        pre_var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print 'Train_Val:',pre_var_list
        var_list=[var for var in pre_var_list if not 'fc6' in var.name]
        #print 'Abandoned:',[var for var in pre_var_list if 'fc6' in var.name]
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss,var_list=var_list)

        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        pre_var_list=tf.global_variables()
        var_list=[var for var in pre_var_list if (not 'fc6-2' in var.name) and (not 'sflabel' in var.name) and (not 'center' in var.name)]
        self.saver2=tf.train.Saver(var_list, max_to_keep=None)



        
    

if __name__ == '__main__':
    # In[57]:

    tf.reset_default_graph()
    model = AFJT(Config)

    # In[36]:



    # In[58]:

    # Train_Loss = agent.register({'Type':'Loss', 'Train_or_Test':'Train', 'initializer':'orthogonal'}, 'loss', overwrite = False)
    # Train_Acc = agent.register({'Type':'Accuracy', 'Train_or_Test':'Train', 'initializer':'orthogonal'}, 'acc', overwrite= False)
    # Train_ED = agent.register({'Type':'Edit_Error', 'Train_or_Test':'Train', 'initializer':'orthogonal'}, 'err', overwrite=False)
    # Test_Loss = agent.register({'Type':'Loss', 'Train_or_Test':'Test', 'initializer':'orthogonal'}, 'loss', overwrite = False)
    # Test_Acc = agent.register({'Type':'Accuracy', 'Train_or_Test':'Test', 'initializer':'orthogonal'}, 'acc', overwrite= False)
    # Test_ED = agent.register({'Type':'Edit_Error', 'Train_or_Test':'Test', 'initializer':'orthogonal'}, 'err', overwrite=False)


    # In[ ]:



    Sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    Sess_config.gpu_options.allow_growth = True
    with tf.Session(config=Sess_config) as sess:

        tf.global_variables_initializer().run()

        start = 0
        if start != 0:
            model.saver.restore(sess=sess, save_path='Model2/AFJT-' + str(start))
        else:
            model.saver2.restore(sess=sess,save_path='Model3/AFJT-200000')
        graph=tf.get_default_graph()
        average_loss = 0.0
        average_loss1 = 0.0
        average_loss2 = 0.0
        average_loss3 = 0.0
        average_acc1 = 0.0
        average_acc2 = 0.0

        import time
        ttt=time.time()
        logw=open('logcacd24.log','w')
        for i in range(start, 500000):

            x_batch, y1_batch,y2_batch= loader.get_batch(i % loader.batch_num)
            # x_batch, y_batch, mask, seq_len=get_m_train_batch(Config.batch_size)

            feed_dict = {model.images: x_batch, model.label1: y1_batch, model.label2: y2_batch,model.training: True}
            loss,loss1,loss2,loss3, acc1,acc2, _ = sess.run([model.loss, model.softmax_loss1,model.softmax_loss2,model.center_loss,model.accuracy1,model.accuracy2,model.train_op], feed_dict=feed_dict)
            #print feat.shape
            

            average_loss = average_loss + (loss / 100)
            average_loss1 = average_loss1 + (loss1 / 100)
            average_loss2 = average_loss2 + (loss2 / 100)
            average_loss3 = average_loss3 + (loss3 / 100)
            average_acc1 = average_acc1 + (acc1 / 100)
            average_acc2 = average_acc2 + (acc2 / 100)
            
            
            if (i) % 100 == 0:
                print(i, ' loss ', average_loss,average_loss1,average_loss2, average_loss3,' acc1 ', average_acc1 * 100.0,' acc2 ', average_acc2 * 100.0)
                logw.write(str(i)+ ' loss '+str( average_loss)+' loss1 '+str(average_loss1)+' loss2 '+str(average_loss2)+' loss3 '+str(average_loss3)+' acc1 '+str(average_acc1 * 100.0)+' acc2 '+str(average_acc2 * 100.0)+'\n')
                
                average_loss = 0.0
                average_loss1 = 0.0
                average_loss2 = 0.0
                average_loss3 = 0.0
                average_acc1 = 0.0
                average_acc2 = 0.0
                print 'time:',time.time()-ttt
                ttt=time.time()
            if (i+1) %200000==0:
                maxBatch=loader_all.batch_num
                maxSample=loader_all.sample_num
                print maxBatch,maxSample
                ori_feat=[]
                ori_y1b=[]
                ori_y2b=[]
                tt=time.time()
                for j in xrange(0,maxBatch):
                    newBatch,y1b,y2b=loader_all.get_batch(j)
                    newFeat=sess.run([model.fc5o],feed_dict={model.images:newBatch,model.label1:y1b,model.label2:y2b,model.training:False})
                    newFeat=np.squeeze(newFeat,axis=(0))
                    newFeat=np.asarray(newFeat)
                    #print newFeat.shape
                    if j==maxBatch-1:
                        newFeat=newFeat[0:maxSample-(maxBatch-1)*Config.batch_size]
                        newFeat=np.asarray(newFeat)
                        #print 'last shape:', newFeat.shape
                        y1b=y1b[0:maxSample-(maxBatch-1)*Config.batch_size]
                        y2b=y2b[0:maxSample-(maxBatch-1)*Config.batch_size]
                        #print y1b,y2b
                    ori_feat.append(newFeat)
                    ori_y1b.append(y1b)
                    ori_y2b.append(y2b)
                    if j%500==0:
                        print 'extract ',j,'batches,time:',time.time()-tt
                train_feat=np.concatenate(ori_feat,axis=0)
                train_ide=np.concatenate(ori_y1b,axis=0)
                train_aux=np.concatenate(ori_y2b,axis=0)
                #print train_feat.shape,train_ide.shape,train_aux.shape
                train_feat=np.transpose(train_feat)
                np.savez('trans.npz',train_feat=train_feat,train_ide=train_ide,train_aux=train_aux)
                F,G,sigma,beta=fa.fa(train_feat,train_ide,train_aux)
                np.savez('transout.npz',F=F,G=G,sigma=sigma,beta=beta)
                print F.shape,G.shape,sigma,beta.shape
            #if (i)%200000==0:
                sss=np.load('transout.npz')
                F=sss['F']
                G=sss['G']
                sigma=sss['sigma']
                beta=sss['beta']
                phi_1=np.linalg.inv(np.dot(F,F.T)+np.dot(G,G.T)+sigma*np.eye(512))
                new_fc6_1w=np.transpose(np.dot(np.dot(F,F.T),phi_1))
                new_fc6_2w=np.transpose(np.dot(np.dot(G,G.T),phi_1))
                new_fc6_1b=-np.dot(np.dot(np.dot(F,F.T),phi_1),beta)
                new_fc6_2b=-np.dot(np.dot(np.dot(G,G.T),phi_1),beta)
                new_fc6_1b=np.squeeze(new_fc6_1b)
                new_fc6_2b=np.squeeze(new_fc6_2b)
                #print new_fc6_1w.shape,new_fc6_1b.shape,new_fc6_2b.shape,new_fc6_2w.shape
                sess.run(tf.assign(graph.get_tensor_by_name('fc6-1/weights:0'),new_fc6_1w))
                sess.run(tf.assign(graph.get_tensor_by_name('fc6-2/weights:0'),new_fc6_2w))
                sess.run(tf.assign(graph.get_tensor_by_name('fc6-1/biases:0'),new_fc6_1b))
                sess.run(tf.assign(graph.get_tensor_by_name('fc6-2/biases:0'),new_fc6_2b))
                #print sess.run(graph.get_tensor_by_name('fc6-1/weights:0')).shape,sess.run(graph.get_tensor_by_name('fc6-1/biases:0'))
                model.ratio=0
            if (i-1000)%200000==0:
                model.ratio=0.008#avoid to the NaN error with center loss when finetune with factor analysis

            if (i) % 2000 == 0 and i != start:
                model.saver.save(sess, 'Modelcacdplus/AFJT', global_step=i )
                print('save model')
        logw.close()


        

