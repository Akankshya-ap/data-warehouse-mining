# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:33:55 2018

@author: Student
"""

import math
import pandas as pd
import numpy as np

filename='D:/115cs0231/28AUG/Example.csv'
df=pd.read_csv(filename)
print df, '\n\n'



df['f1+']=df['f11']+df['f10']
df['f0+']=df['f01']+df['f00']
df['f+1']=df['f11']+df['f01']
df['f+0']=df['f00']+df['f10']
df['N']=df['f1+']+df['f0+']
print df , '\n\n'


df['Correlation']=(df['N']*df['f11']-df['f1+']*df['f+1'])/np.sqrt((df['f1+']*df['f+1']*df['f0+']*df['f+0']))
df['Odds-ratio']=df['f11']*df['f00']/(df['f10']*df['f01'])
df['Kappa']=(df['N']*df['f11']+df['N']*df['f00']-df['f1+']*df['f+1']-df['f0+']*df['f+0'])/(df['N']*df['N']-df['f+1']*df['f1+']-df['f0+']*df['f+0'])
df['Interest']=df['N']*df['f11']/(df['f1+']*df['f+1'])
df['Piatetsky shapiro']=df['f11']/df['N']-df['f1+']*df['f+1']/(df['N']*df['N'])
df['Collection-strength']=((df['f11']+df['f00'])/(df['f1+']*df['f+1']+df['f0+']*df['f+0']))*((df['N']-df['f1+']*df['f+1']-df['f0+']*df['f+0'])/(df['N']-df['f11']-df['f00']))
df['Jaccard']=df['f11']/(df['f1+']*df['f+1']-df['f11'])
#df['allconf']=np.ndarray.min((df['f11']/df['f1+']),(df['f11']/df['f+1']))


print df