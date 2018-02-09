import numpy as np
from numpy.linalg import inv
import time

def fa(feat,ide,aux,p=500,q=50):
    d=feat.shape[0]
    print "d",d
    Id=np.eye(d)
    Ip=np.eye(p)
    Iq=np.eye(q)
    N=2000
    K=5046
    maxEpoch=10
    T1=np.zeros((d,N))
    T2=np.zeros((d,K))
    T3=np.asarray(feat)
    print 't3.shape:',T3.shape
    Ni=np.zeros((1,N))
    Mk=np.zeros((1,K))
    print 'Ni.shape:',Ni.shape,Ni[0].shape
    beta=np.expand_dims(np.mean(T3,1),1)
    print beta.shape
    T30=T3-beta
    T3T=T30.T
    tt=time.time()
    for i in range(0,N):
        tmp=[]
        
        for pp in range(0,T30.shape[1]):
            if ide[pp]==i:
                tmp.append(T3T[pp])
        tmp=np.asarray(tmp)
        if i==0:
            print 'tmp.shape:',tmp.shape
        tmp=tmp.T
        T1[:,i]=np.mean(tmp-beta,1)
        Ni[0,i]=tmp.shape[1]
        #if i%100==0:
        #	print 'N',i,':',Ni[0,i]
    for k in range(0,K):
        tmp=[]
        for pp in range(0,T30.shape[1]):
            if aux[pp]==k:
                tmp.append(T3T[pp])
        tmp=np.asarray(tmp)
        if k==0:
            print 'tmp.shape:',tmp.shape
        tmp=tmp.T
        T2[:,k]=np.mean(tmp-beta,1)
        Mk[0,k]=tmp.shape[1]
    print 'time1:',time.time()-tt
    print T1.shape,T2.shape
    tt=time.time()

    G=np.zeros((K,N))
    for pp in range(0,T30.shape[1]):
        G[aux[pp],ide[pp]]=G[aux[pp],ide[pp]]+1
    MGN=np.dot(np.dot((1/Mk)**0.5,G),np.transpose((1/Ni)**0.5))
    #print MGN

    sigma=0.1
    U=0.2*(np.random.rand(d,p)-0.5)
    V=0.2*(np.random.rand(d,q)-0.5)

    for itern in range(0,maxEpoch):
        S=sigma*Id+np.dot(U,U.T)+np.dot(V,V.T)
        Ex=np.dot(np.dot(U.T,inv(S)),T1)
        Ey=np.dot(np.dot(V.T,inv(S)),T2)
        A=(Ip-np.dot(np.dot(U.T,inv(S)),U))*N+np.dot(np.dot(Ex,np.diag(Ni[0])),Ex.T)
        B=(Iq-np.dot(np.dot(V.T,inv(S)),V))*N+np.dot(np.dot(Ey,np.diag(Mk[0])),Ey.T)
        C=np.dot(T1*Ni,Ex.T)
        D=np.dot(T2*Mk,Ey.T)
        E=-np.dot(np.dot(V.T,inv(S)),U)*MGN+np.dot(np.dot(Ey,G),Ex.T)
        F=np.transpose(E)
        Ux=np.dot(U,Ex)
        Vy=np.dot(V,Ey)
        U=np.dot(C-np.dot(np.dot(D,inv(B)),E),inv(A-np.dot(np.dot(F,inv(B)),E)))
        V=np.dot(D-np.dot(np.dot(C,inv(A)),F),inv(B-np.dot(np.dot(E,inv(A)),F)))
        T4=np.zeros((d,T30.shape[1]))
        for pp in range(0,T30.shape[1]):
            T4[:,pp]=Ux[:,ide[pp]]+Vy[:,aux[pp]]
        sigma=np.sum((T30-T4)*T30)/(d*T30.shape[1])
        print 'itern:',itern,'timecost:',time.time()-tt,'sigma:',sigma
        tt=time.time()

    return U,V,sigma,beta


def meanNorm(X,meanVec=None):
    if meanVec==None:
        m=np.mean(X,1)
    else:
        m=meanVec
    Y=X-m
    return Y


