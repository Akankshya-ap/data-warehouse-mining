# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:08:30 2018

@author: Student
"""

filename='D:/115cs0231/8.18sep/iris.csv'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy

ds=pd.read_csv(filename)
ds["flower"] = pd.Categorical(ds["flower"])
ds["flower"] = ds["flower"].cat.codes
data = ds.values[:, 0:4]
category = ds.values[:, 4]
ds=ds[['sw','sl','pw','pl']]
print (ds.head())
from scipy.cluster.hierarchy import dendrogram, linkage

# generate the linkage matrix
Z = linkage(ds, 'ward')

# set cut-off to 150
max_d = 7.08                # max_d as in max_distance

plt.figure(figsize=(10, 10))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=150,                  # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=8.,      # font size for the x axis labels
)
plt.axhline(y=max_d, c='k')
plt.show()



colors=['blue', 'orange', 'green']
for i in range(150):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])