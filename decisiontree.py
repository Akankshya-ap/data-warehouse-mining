# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 14:05:35 2018

@author: Student
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

filename='D:/115cs0231/4sep/car.csv'
df=pd.read_csv(filename)
print df.head(), '\n\n'

print 'The Attributes are:' ,'\n' ,df.describe()

print df.shape[0]

print "dfset Length:: ", len(df)
print "dfset Shape:: ", df.shape

df['class'],class_names = pd.factorize(df['class'])
df['buying'],_ = pd.factorize(df['buying'])
df['maint'],_ = pd.factorize(df['maint'])
df['doors'],_ = pd.factorize(df['doors'])
df['person'],_ = pd.factorize(df['person'])
df['lug_boot'],_ = pd.factorize(df['lug_boot'])
df['safety'],_ = pd.factorize(df['safety'])

print df

X = df.values[:, 0:6]
Y = df.values[:,-1]
print X


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)
print X_train.shape


clf_gini = DecisionTreeClassifier(criterion = "gini", splitter='best' ,random_state = 0)
                     
clf_gini.fit(X_train, y_train)

'''DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
'''
#clf_gini.predict([[3, 3, 2, 1, 0, 2]])

y_pred = clf_gini.predict(X_test)
print y_pred, y_test
print accuracy_score(y_test,y_pred)*100