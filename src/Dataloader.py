import numpy as np
import cv2
import random

#import self as self


class Dataloader():
    def __init__(self, batch_size, gtfile='trainCASIA.txt'):

        file = open(gtfile, 'r')
        # rootfolder = "/data1/home/chunchaoguo/data/bankrealtrain/"
        lines = file.readlines()

        random.shuffle(lines)
        self.sample_num = len(lines)
        self.batch_size = batch_size
        self.batch_num = self.sample_num // batch_size
        self.pairs = [line.split() for line in lines]

        #self.index2char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        #                   'STA', 'END', 'PAD']
        #self.char2index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        #                   '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        #                   'STA': 10, 'END': 11, 'PAD': 12}
#        self.max_len = max_len
#        self.dict_file = dict_file
#        self.__build_dict()
#
#    def __build_dict(self):
#        self.word_dict = []
#        for w in open(self.dict_file, 'r'):
#            # print(w)
#            chr = w.split()[1].strip()
#            self.word_dict.append(chr.decode('utf-8'))
#        self.word_dict.append('STA'.decode('utf-8'))
#        self.word_dict.append('END'.decode('utf-8'))
#        self.word_dict.append('PAD'.decode('utf-8'))
#
#    def get_dict(self):
#        return self.word_dict
#
#    def get_label(self,s):
#        try:
#            i = self.word_dict.index(s)
#        except ValueError:
#            print 'Charictor not in dict:',s.encode('utf-8')
#            return -1
#        return i

    def get_batch(self, batch_index, img_size=[112, 96],get_path=False):
        if batch_index==0:
            random.shuffle(self.pairs)
        batch_pairs = self.pairs[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        batch_img = []
        batch_label = []


        for path, labelstr in batch_pairs:

            temp_img = cv2.imread(path)
            temp_img = cv2.resize(temp_img, (img_size[1],img_size[0]),
                                  interpolation=cv2.INTER_CUBIC)
            temp_img = (temp_img -127.5)/128
            batch_img.append(temp_img)

            #temp_txt = [self.char2index[c] for c in txt]
            #temp_txt=[self.get_label(c) for c in txt.decode('utf-8')]
            
            #temp_txt = temp_txt + [self.char2index['END']] + [self.char2index['PAD']] * pad_len
            temp_label = int(labelstr)
            temp_label = np.array(temp_label)
            batch_label.append(temp_label)
        #print temp_img.shape,temp_label.shape
        batch_img = np.array(batch_img)
        #batch_img = np.expand_dims(batch_img, 3)
        batch_label = np.array(batch_label)


        return batch_img, batch_label


if __name__ == '__main__':
    loader = Dataloader(batch_size=64)

    for i in range(5):
        batch_img, batch_label = loader.get_batch(i)
        print batch_img.shape, batch_label.shape
        print batch_label
        print batch_img[0]

    print('Good!')
