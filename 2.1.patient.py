# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 13:52:08 2018

@author: Student
"""

import numpy as np
import math
import pandas as pd
import csv

dfile='D:/115cs0231/patient.csv'
#df=csv.reader(dfile)
df=pd.read_csv(dfile)
print "\n\nFirst five rows of the dataset "
print (df.head())


df['Gender']=df['Gender'].map({'female':'1','male':'0'})
print "\n\nConverting male to 0 and female to 1"
print df


df['HasCancer']=df['HasCancer'].map({True:'1',False:'0'})
print "\n\nWhether he has cancer or not"
print df

#av_age=np.average(df['Age'])
df['Age']=df['Age'].fillna(df['Age'].mean()).astype(int)
print "\n\nFilling age"
print df

tumormean=df.groupby('Age')['Tumorsize'].mean()
#print tumormean

print "\n\nAssigning tumorsize to blank data"
for i,c in enumerate(df['Tumorsize']):
    #print i
    if math.isnan(c)==1:
        #print 'a'
        #print df['Age'][i]
        agegroup=df['Age'][i]
        x=tumormean[agegroup]
        #print x
        df['Tumorsize'][i]=x
        #print df['Age']


print df
dfile2='D:/115cs0231/patient2.csv'
df.to_csv(dfile2, sep=',', encoding='utf-8', index=False)


#df['Tumorsize']=df['Tumorsize'].fillna()