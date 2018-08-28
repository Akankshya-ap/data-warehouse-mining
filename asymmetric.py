import pandas as pd
import numpy
df=pd.read_csv("D:/115cs0231/28AUG/Example.csv")
print df, '\n\n'
df['f1+'] = df['f11'] + df['f10']
df['f0+'] = df['f01'] + df['f00']
df['f+1'] = df['f11'] + df['f01']
df['f+0'] = df['f10'] + df['f00']
df['N'] = df["f1+"] + df["f0+"]
df['Laplace'] = (df["f11"]+1)/(df['f1+']+2)
df['Conviction'] = (df["f1+"]*df["f+0"])/(df['N']*df['f10'])
df['Added value'] = (df['f11']/df['f1+'])-(df['f+1']/df['N'])
df['Certainty factor'] = ((df['f11']/df['f1+'])-(df['f+1']/df['N']))/(1-(df['f1+']/df['N']))
df['Gini index'] = (df['f1+']/df['N'])*((df['f11']/df['f1+'])*(df['f11']/df['f1+'])+(df['f10']/df['f1+'])*(df['f10']/df['f1+']))-((df['f+1']/df['N'])*(df['f+1']/df['N']))+(df['f0+']/df['N'])*((df['f01']/df['f0+'])*(df['f01']/df['f0+'])+(df['f00']/df['f0+'])*(df['f00']/df['f1+']))-((df['f+0']/df['N'])*(df['f+0']/df['N']))
df['J Measure'] = (df['f11']/df['N'])*numpy.log2((df['N']*df['f11'])/(df['f+1']*df['f1+']))+(df['f10']/df['N'])*numpy.log2((df['N']*df['f10'])/(df['f1+']*df['f+0']))
df['Mutual Information'] = ((df['f11']/df['N'])*numpy.log2(df['f11']/(df['f1+']*df['f+1']))+(df['f10']/df['N'])*numpy.log2(df['f10']/(df['f1+']*df['f+0']))+(df['f01']/df['N'])*numpy.log2(df['f01']/(df['f0+']*df['f+1']))+(df['f00']/df['N'])*numpy.log2(df['f00']/(df['f0+']*df['f+0'])))-((df['f1+']/df['N']*numpy.log2(df['f1+']/df['N']))+(df['f0+']/df['N']*numpy.log2(df['f0+']/df['N'])))
print df