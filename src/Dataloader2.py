import numpy as np
import cv2
import random

#import self as self


class Dataloader():
    def __init__(self, batch_size, gtfile='trainCACD.txt'):

        file = open(gtfile, 'r')
        # rootfolder = "/data1/home/chunchaoguo/data/bankrealtrain/"
        lines = file.readlines()

        random.shuffle(lines)
        self.sample_num = len(lines)
        self.batch_size = batch_size
        self.batch_num = self.sample_num // batch_size
        self.pairs = [line.split() for line in lines]


    def get_batch(self, batch_index, img_size=[112, 96],get_path=False):
        if batch_index==0:
            random.shuffle(self.pairs)
        batch_pairs = self.pairs[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        batch_img = []
        batch_label1 = []
        batch_label2 = []


        for path, labelstr1,labelstr2 in batch_pairs:

            temp_img = cv2.imread(path)
            temp_img = cv2.resize(temp_img, (img_size[1],img_size[0]),
                                  interpolation=cv2.INTER_CUBIC)
            temp_img = (temp_img -127.5)/128
            batch_img.append(temp_img)

            #temp_txt = [self.char2index[c] for c in txt]
            #temp_txt=[self.get_label(c) for c in txt.decode('utf-8')]
            
            #temp_txt = temp_txt + [self.char2index['END']] + [self.char2index['PAD']] * pad_len
            temp_label1 = int(labelstr1)
            temp_label1 = np.array(temp_label1)
            batch_label1.append(temp_label1)
            temp_label2 = int(labelstr2)
            temp_label2 = np.array(temp_label2)
            batch_label2.append(temp_label2)
        #print temp_img.shape,temp_label.shape
        batch_img = np.array(batch_img)
        #batch_img = np.expand_dims(batch_img, 3)
        batch_label2 = np.array(batch_label2)
        batch_label1 = np.array(batch_label1)


        return batch_img, batch_label1,batch_label2
        

class Dataloader_all():
    def __init__(self, batch_size, gtfile='trainCACD.txt'):

        file = open(gtfile, 'r')
        # rootfolder = "/data1/home/chunchaoguo/data/bankrealtrain/"
        lines = file.readlines()
        self.sample_num = len(lines)
        self.batch_size = batch_size
        self.batch_num = self.sample_num // batch_size

        self.pairs = [line.split() for line in lines]
        if self.sample_num>self.batch_num*self.batch_size:
            self.batch_num=self.batch_num+1
            for i in range(0,self.batch_size*self.batch_num-self.sample_num):
                self.pairs.append(['0','0','0'])


    def get_batch(self, batch_index, img_size=[112, 96],get_path=False):
        batch_pairs = self.pairs[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        batch_img = []
        batch_label1 = []
        batch_label2 = []


        for path, labelstr1,labelstr2 in batch_pairs:
            if path=='0':
                temp_img=np.zeros((112,96,3))
            else:
                temp_img = cv2.imread(path)
                temp_img = cv2.resize(temp_img, (img_size[1],img_size[0]),
                                      interpolation=cv2.INTER_CUBIC)
                temp_img = (temp_img -127.5)/128
            batch_img.append(temp_img)

            #temp_txt = [self.char2index[c] for c in txt]
            #temp_txt=[self.get_label(c) for c in txt.decode('utf-8')]
            
            #temp_txt = temp_txt + [self.char2index['END']] + [self.char2index['PAD']] * pad_len
            temp_label1 = int(labelstr1)
            temp_label1 = np.array(temp_label1)
            batch_label1.append(temp_label1)
            temp_label2 = int(labelstr2)
            temp_label2 = np.array(temp_label2)
            batch_label2.append(temp_label2)
        #print temp_img.shape,temp_label.shape
        batch_img = np.array(batch_img)
        #batch_img = np.expand_dims(batch_img, 3)
        batch_label2 = np.array(batch_label2)
        batch_label1 = np.array(batch_label1)


        return batch_img, batch_label1,batch_label2
        


if __name__ == '__main__':
    loader = Dataloader(batch_size=64)

    for i in range(5):
        batch_img, batch_label1,batch_label2 = loader.get_batch(i)
        print batch_img.shape, batch_label1.shape,batch_label2.shape
        print batch_label1,batch_label2
        print batch_img[0]

    print('Good!')
