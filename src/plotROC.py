import os
import cv2
import sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize']=24
mpl.rcParams['ytick.labelsize']=24

def main():
    gto=open('logcacdvs207.log','r')
    lines=gto.readlines()
    dis=[]
    for line in lines:
        dis.append(float(line.split(' ')[1]))
    print(len(dis),dis[0])
    sot=sorted(dis)
    fpr=[]
    far=[]
    for testi in range(0,4000):
        fenge=sot[testi]
        label=[]
        for distance in dis:
    	    if distance>fenge:
    		    label.append(0)
    	    else:
    		    label.append(1)
    #print label
        fp=0
        fa=0
        for i in xrange(0,10):
    	    for j in xrange(i*200,(i+1)*200):
    		    if i in [0,2,4,6,8]:
    			    if label[j]==0:
    				    fp+=1
    		    if i in [1,3,5,7,9]:
    			    if label[j]==1:
    				    fa+=1
        if testi%10==0:
            print testi,', fpr:',fp/2000.0,'far:', fa/2000.0
        far.append(fa/2000.0)
        fpr.append(fp/2000.0)
    fpr=np.asarray(fpr)
    far=np.asarray(far)
    plt.figure('ROC')
    plt.plot(far,1-fpr)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()





if __name__ == '__main__':
    main()