# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 15:17:45 2018

@author: Student
"""

import numpy as np
import math

x=np.array(np.random.rand(10))
y=np.array(np.random.rand(10))
print x,y

def euclideandis(x,y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    
print "\nEuclidean distance" ,euclideandis(x,y)

def manhattan_distance(x,y):
 
    return sum(abs(a-b) for a,b in zip(x,y))

print "\nManhattan distance ",manhattan_distance(x,y)


from decimal import Decimal
 
def nth_root(value, n_root):
 
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)
 
def minkowski_distance(x,y,p_value):
 
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)
 
print "\nMinkowski distance ",minkowski_distance(x,y,5)