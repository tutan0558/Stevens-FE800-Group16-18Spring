# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:39:07 2018

@author: 79127
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.optimize import bisect

from numba import jit

plt.style.use('seaborn')
'''
data = pd.read_csv('800data.csv')
data.drop('XLRE', axis = 1, inplace = True)
data.set_index('Date', inplace = True)
data.index = pd.to_datetime(data.index)
data.info()

XLRE = pd.read_csv('XLRE.csv')
XLRE.set_index('Date', inplace = True)
XLRE.index = pd.to_datetime(XLRE.index)
XLRE.info()

data = data.join(XLRE['Close'])
data.rename(columns={'Close': 'XLRE'}, inplace = True)
data.to_csv('800FinalData.csv')
'''
pd.read_csv?
data = pd.read_csv('800FinalData.csv',index_col = 0, parse_dates = True)
data.info()
ETFs = data.columns

Return = pd.DataFrame()

for ETF in ETFs: 
    Return["{}_Return".format(ETF)] = data[ETF].pct_change()
  

def Var(array, confidence_interval=0.95): 
    # temp = list(array)
    temp = np.sort(array)
    
    return temp[round(len(temp)*(1-confidence_interval))-1]

Var_100Day = pd.DataFrame()
Var_500Day = pd.DataFrame()
Var_50Day = pd.DataFrame()
Var_10Day = pd.DataFrame()


for ETF in data.columns: 
    Var_10Day[ETF] = Return['{}_Return'.format(ETF)].rolling(10).apply(Var)
    Var_50Day[ETF] = Return['{}_Return'.format(ETF)].rolling(50).apply(Var)
    Var_100Day[ETF] = Return['{}_Return'.format(ETF)].rolling(100).apply(Var)
    Var_500Day[ETF] = Return['{}_Return'.format(ETF)].rolling(500).apply(Var)
   
Var_500Day.hist(bins = 20, figsize = (15,10))
Var_50Day.hist(bins = 20, figsize = (15,10))
Var_100Day.hist(bins = 20, figsize = (15,10))
Var_10Day.hist(bins = 20, figsize = (15, 10))

Var_100Day.SPY.hist(bins = 50, figsize = (7,10))

Var_10Day.plot(figsize = (15,10))

SPY100Var1 = Var_100Day.SPY
SPY100Var1.dropna(inplace = True)

Var_100Day.plot(figsize = (15,10), grid = True)
Var_10Day.info()


Test = Var_100Day.dropna()


def f(x, s, k):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-(x**2)/2) * (1+(s/6)*(x**3 - 3*x) + k/24 * (x**4 - 6*x**2 + 3))


def integrate(b):
    x = np.linspace(-200, b, 1000000)
    fx = f(x, S, K)
    area = np.sum(fx)*(b+200)/1000000
    return area -0.95

jit?
GC_Var = pd.DataFrame()

Log_Return = pd.DataFrame()

for ETF in ETFs: 
    Log_Return["{}_Log_Return".format(ETF)] = np.log(data[ETF] / data[ETF].shift(1))


Log_Return.reset_index(inplace = True)


for i in np.arange(len(Log_Return)-100): 
    temp = Log_Return.loc[i:i+100, 'SPY_Log_Return']
    S = skew(temp)
    K = kurtosis(temp)
    GC_Var.loc[i, 'SPY'] = bisect(integrate, -5,5)
    
integrate(1000)
    
bisect(integrate, -5,5)

Log_Return.loc[5:5+100, 'SPY_Log_Return']

GC_Var.to_csv('GC_SPY_VaR.csv')

plt.subplot(1, 2,1)

plt.plot(GC_Var['SPY']/(-100))
plt.subplot(1, 2,2)
plt.plot(Var_100Day['SPY'])

Var_100Day['SPY'].hist(bins = 30, figsize = (15, 10))


np.var(GC_Var['SPY']/100*(-1))
np.var(Var_100Day['SPY'])
GC_Var.drop(0, inplace = True)
GC_Var.hist(bins = 30, figsize = (15, 10))    
    
GC_Var['EXP'] = np.exp(GC_Var['SPY'])
plt.plot(Var_100Day.iloc[:236,0])
plt.plot(GC_Var['EXP'])



kurtosis?






