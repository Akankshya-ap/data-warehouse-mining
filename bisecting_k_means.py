# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:45:23 2018

@author: Student
"""

filename='D:/115cs0231/8.18sep/iris.csv'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from math import sqrt, floor
import math
from copy import deepcopy

ds=pd.read_csv(filename)
#print ds
#ds['flower'],class_names = pd.factorize(ds['flower'])
ds["flower"] = pd.Categorical(ds["flower"])
ds["flower"] = ds["flower"].cat.codes


# Change dataframe to numpy matrix
data = ds.values[:, 0:4]
category = ds.values[:, 4]
# Number of clusters
k = 2
k_=3
# Number of training data
n = data.shape[0]
# Number of features in the data
c = data.shape[1]


# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = []
# Plot the data and the centers generated as random
colors=['blue', 'orange', 'green']
#for i in range(n):
#    plt.scatter(data[i, 0], data[i,1], s=5, color = colors[int(category[i])])
#plt.scatter(centers[:,0], centers[:,1], marker='*', c='red', s=150)

#centers_old = np.zeros(centers.shape) # to store old centers
#centers_new = deepcopy(centers) # Store new centers
#print centers_new
centroid1=np.mean(data,axis=0)
print centroid1
print data.shape
centers.append(centroid1)
print len(centers)
clusters = np.zeros(n)
#distances = np.zeros((n,k))


def sse(dat,centroid,length):
    dis=0
    #print len(dat[0])
    #print centroid
    for x in range(len(dat[0])):
        dis+=(dat[0][x]-centroid)**2
    return dis
#error = np.linalg.norm(centers_new - centers_old)
itera=0
clu=1
data_new=[]
data_new.append(data)
t1=np.where(clusters==0)
data1=[data[t1]]
#print data1
#print len(data_new)
#print clusters
# When, after an update, the estimate of that center stays the same, exit loop
while(clu!=k_):
    #clu=clu+1
    for a in range(0,len(centers)):
        
        #print (data1)
        print "total data",len(data1[a])
        sse_p=sse(data1,centers[a],4)
        print "parent sse", sse_p
        mean = np.mean(data1[a], axis = 0)
        std = np.std(data1[a], axis = 0)
        c = data1[a].shape[1]
        #print c
        centers_r = np.random.randn(k,c)*std + mean
        centers_new = deepcopy(centers_r)
        #print centers_new
        itera=0
        distances = np.zeros((data1[a].shape[0],2))
        #print distances
        while itera != 50:
            # Measure the distance to every center
            for i in range(2):
                distances[:,i] =np.linalg.norm(data1[a] - centers_new[i], axis=1)
            #print (distances[:,i])
            # Assign all training data to closest center
            clusters_r = np.argmin(distances, axis = 1)
            #print clusters
            #print len(data[clusters_r==0])
            #print len(data[clusters_r==1])
            centers_old = deepcopy(centers_new)
            # Calculate mean for every cluster and update the center
            for i in range(k):
                t1=np.where(clusters_r==i)
                centers_new[i] = np.mean(data1[a][t1], axis=0)
                
            error = np.linalg.norm(centers_new - centers_old)
            itera=itera+1
        #print clusters_r
        c1=np.where(clusters_r==0)
        c2=np.where(clusters_r==1)
        #print c2, error
        #print len(distances)
        data11= [data1[a][c1]]
        #print (data11)
        #print np.where(data[clusters_r==0])
        data12=  [data1[a][c2]]
        print "child1",len(data11[0])
        print "child2", len(data12[0])
        sse_c1=sse(data11,centers_new[0],4)
        sse_c2=sse(data12,centers_new[1],4)
        print sse_c1
        print sse_c2
        if(sse_c1.all<=sse_p.all or sse_c2.all<=sse_p.all):
            centers.remove(centers[a])
            #print centers
            centers.append(centers_new[0])
            centers.append(centers_new[1])
            
            data1.remove(data1[a])
            #print data1
            data1.append(data11[0])
            data1.append(data12[0])
            #print len(data1)
            clusters=clusters_r
            #data1.remove(data1[])
            clu=clu+1
            print "no of clusters=" ,clu
            if(clu==k_):
                break
    #print "new centers \t \t", centers
                #centers_new    



colors=['blue', 'orange', 'green']
#for i in range(n):
    #plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(clusters)])
#print clusters
#print centers[0]
pq=np.asarray(centers)
print centers
plt.scatter(pq[:,0], pq[:,3], marker='*', c='black', s=200)
print data1[0][:,0]
plt.scatter(data1[0][:,0],data1[0][:,3],c='blue',s=10)
plt.scatter(data1[1][:,0],data1[1][:,3],c='red',s=10)
plt.scatter(data1[2][:,0],data1[2][:,3],c='yellow',s=10)
#plt.scatter(data1[:,0],data1[:])