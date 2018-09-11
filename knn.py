# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:40:19 2018

@author: Student
"""

filename='D:/115cs0231/7.9sep/car.csv'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs


ds=pd.read_csv(filename)
#print ds
ds['class'],class_names = pd.factorize(ds['class'])
ds['buying'],_ = pd.factorize(ds['buying'])
ds['maint'],_ = pd.factorize(ds['maint'])
ds['doors'],_ = pd.factorize(ds['doors'])
ds['person'],_ = pd.factorize(ds['person'])
ds['lug_boot'],_ = pd.factorize(ds['lug_boot'])
ds['safety'],_ = pd.factorize(ds['safety'])


###different classes division#####

prop_amt=0.80
print 'train:test '+str(prop_amt)+':'+str(1-prop_amt)

X = ds.values[:, 0:6]
Y = ds.values[:,-1]

values=np.unique(Y)
#print Y
c=len(values)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 40)

train_x=X_train
train_y=Y_train
x=X_test #np.array(X_test,dtype=float)
y=Y_test #np.array(Y_test)

#print y
#print X_train
#print X_test
#print x
#print np.transpose(x)
mean1= np.unique(np.array([Y_test])).reshape(c,1)

#################distance from each#######
import math
def eDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

########getting neighbor########   
import operator 
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    #print length
    for x in range(len(trainingSet)):
        #print trainingSet[x]
        dist = eDistance(testInstance, trainingSet[x], length)
        distances.append((train_y[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        #print distances[x][0]
        ind=np.nonzero(distances[x][0]==values)[0]
        #print ind
        #t=datay.loc[ds.index.isin(ind)]
        #print t
        neighbors.append(ind)
    return neighbors


##########main############
l=[]
knnp=[]
acc=[]
p=50
#print 'k='+str(k)
for knn in range(1,p):
    l=[]
    for  i in range(len(x)):
        #print np.array(X_train)[0]
        nb=getNeighbors(train_x,x[i],knn)
        #print nb
        #print len(nb)
        #print nb[1][-1]
        
        q=[0]*c
        max=0
        for u in range(len(nb)):
            #print 'u'
            for w in range(c):
                #print w
                #print nb[u]
                if nb[u]==w:
                    #print nb[u]
                    q[w]+=1
                    if(q[w]>max):
                        max=q[w]
                        #print max
                        n=w
        #print mean1[n]               
        l.append(mean1[n])           
        
    r=np.array(l)
    
    #######printing fitted values####    
    #print  r
    
    #####printing actual value###
    #print y
    
    
    #####getting accuracy#####
    j=0
    k=0
    
    for i in range(0,len(x),1):
        if r[j]!=y[i]:
            k=k+1
        j=j+1
    accu=(float)(len(x)-k)*100/len(x)
    acc.append(accu)
    knnp.append(knn)
    print 'k', knn
    print 'Accuracy= ' + str(accu)


######plotting#####

print np.shape(knnp)
print np.shape(acc)
plt.plot(knnp, acc, label = "line 1")
plt.show()