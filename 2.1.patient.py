# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 13:52:08 2018

@author: Student
"""

import numpy as np
import pandas as pd
import csv

dfile='D:/115cs0231/patient.csv'
#df=csv.reader(dfile)
df=pd.read_csv(dfile)
print(df.head())
df['Gender']=df['Gender'].map({'female':'1','male':'0'})
print df
df['HasCancer']=df['HasCancer'].map({True:'1',False:'0'})
print df

#av_age=np.average(df['Age'])
df['Age']=df['Age'].fillna(df['Age'].mean()).astype(int)
print df

tumormean=df.groupby('Age')['Tumorsize'].mean()
print tumormean

dfile2='D:/115cs0231/patient2.csv'
df.to_csv(dfile2, sep='\t', encoding='utf-8')

#df['Tumorsize']=df['Tumorsize'].fillna()