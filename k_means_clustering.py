# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 13:43:36 2018

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
k = 3
# Number of training data
n = data.shape[0]
# Number of features in the data
c = data.shape[1]


# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean
print centers
# Plot the data and the centers generated as random
colors=['blue', 'orange', 'green']
#for i in range(n):
#    plt.scatter(data[i, 0], data[i,1], s=5, color = colors[int(category[i])])
plt.scatter(centers[:,0], centers[:,1], marker='*', c='red', s=150)

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers
#print centers_new

print data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

def eDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        #print (instance2[x])
        
        distance += pow((instance1[x] - instance2[x]), 2)
        #print(1)
    return math.sqrt(distance)


error = np.linalg.norm(centers_new - centers_old)
itera=0
# When, after an update, the estimate of that center stays the same, exit loop
while itera != 100:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] =np.linalg.norm(data - centers[i], axis=1)
        print len(distances[:,i])
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
    itera=itera+1
centers_new    



colors=['blue', 'orange', 'green']
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='black', s=150)
