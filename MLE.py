# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:22:32 2018

@author: 79127
"""
import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
minimize?

def LL(params, data): 
    '''
    params is a ndarray, [mean, variance, skew, kurt]
    '''
    mean = params[0]
    sig = params[1]
    skew = params[2]
    kurt = params[3]
    x = data
    norm = 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mean)**2 / (2*sig**2))
    H3 = ((x - mean)/sig)**3 - 3*((x- mean)/sig)
    H4 = ((x - mean)/sig)**4 - 6*((x - mean)/sig)**2 + 3
    temp1 = 1+skew/(6*sig**3)*H3 + (kurt-3)/(24*sig**4)*H4
    temp2 = 1+skew**2/(6*sig**3) + (kurt-3)**2/(24*sig**4)
    f = norm*temp1**2/temp2
    ll = np.sum(np.log(f))
    
    return -ll

def LL(params, data): 
    # Params
    mean = params[0]
    sig = params[1]
    skew = params[2]
    kurt = params[3]
    # Standardize data
    x = (data - mean) / sig
    
    # Compose PDF
    
    norm = 1 / (np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)
    temp1 = 1+skew/6 * (x**3 - 3*x) + (kurt-3)/24 * (x**4 - 6*x**2 + 3)
    temp2 = 1+skew**2 / 6 + (kurt-3)**2 / 24
    f = norm * temp1**2 / temp2
    
    # Log Maximum Likelihood Function
    ll = np.sum(np.log(f))
    
    return -ll
    


def MY_GC(x,params):
    mean = params[0]
    sig = params[1]
    skew = params[2]
    kurt = params[3]
    norm = 1 / (np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)
    temp1 = 1+skew/6 * (x**3 - 3*x) + (kurt-3)/24 * (x**4 - 6*x**2 + 3)
    temp2 = 1+skew**2 / 6 + (kurt-3)**2 / 24
    f = norm * temp1**2 / temp2
    return f
    
MY_GC(1, MLE_result)   
    

x = np.linspace(-7, 7, 10000)
plt.figure(figsize = (15,10))
plt.title('Gram-Charlier With Positive Constraints')
plt.plot(x, MY_GC(x, MLE_result))

SPY100 = pd.read_clipboard()
SPY100.rename(columns = {'V1' : 'SPY_Positive'}, inplace = True)
SPY100.SPY_Positive = SPY100.SPY_Positive/100
SPY100.plot(secondary_y = ('SPY'),)
SPY100.hist(bins = 30, figsize = (15,10))
SPY100.hist(bins = 30, figsize = (7, 10))
SPY100.info()
SPY100

SPY100.info()


SPY100Var1 = SPY100Var1.reset_index()

Var_100Day.reset_index(inplace = True)
SPY100 = SPY100.join(SPY100Var1.SPY)
SPY100.info()

SPY100.SPY_Positive = SPY100.SPY_Positive/100



mean, sig, skew, kurt = np.array([0,1,1,4])


LL(np.array([1,1,1,3]), spy)

spy = np.log(data.SPY / data.SPY.shift(1))

x = spy
MLE_result = minimize(LL, x0=np.array([0,1,1,4]), args=spy, method = 'L-BFGS-B', bounds = ((-0.5, 0.5), (0.5, 1.1), (-200, 200), (0, 10)))['x']

minimize(LL, x0=np.array([0,1,1,4]), args=spy, method = 'L-BFGS-B', bounds = ((-0.5, 0.5), (0.5, 1.1), (-200, 200), (0, 10)))


def Positive(x, params):
    mean = params[0]
    sig = params[1]
    skew = params[2]
    kurt = params[3]
    norm = 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mean)**2 / (2*sig**2))
    H3 = ((x - mean)/sig)**3 - 3*((x - mean)/sig)
    H4 = ((x- mean)/sig)**4 - 6*((x - mean)/sig)**2 + 3
    temp1 = 1+skew/(6*sig**3)*H3 + (kurt-3)/(24*sig**4)*H4
    temp2 = 1+skew**2/(6*sig**3) + (kurt-3)**2/(24*sig**4)
    return norm*temp1**2/temp2






x = 5

mean = MLE_result[0]
sig = MLE_result[1]
skew = MLE_result[2]
kurt = MLE_result[3]
norm = 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mean)**2 / (2*sig**2))
H3 = ((data - mean)/sig)**3 - 3*((data - mean)/sig)
H4 = ((data - mean)/sig)**4 - 6*((data - mean)/sig)**2 + 3
temp1 = 1+skew/(6*sig**3)*H3 + (kurt-3)/(24*sig**4)*H4
temp2 = 1+skew**2/(6*sig**3) + (kurt-3)**2/(24*sig**4)
norm*temp1**2/temp2



Positive(1, MLE_result)

x = np.linspace(-0.03, 0.03, 100000)
plt.plot(x, Positive(x, MLE_result))



ss = pd.read_csv('ss.csv')
ss

def integrate(b):
    x = np.linspace(-10, b, 1000000)
    fx = Positive(x, params)
    area = np.sum(fx)*(b+10)/1000000
    return area -0.05

Positive(2, params)
integrate(2)
params = np.array(ss.iloc[1,1:5])
bisect(integrate, -20,-2)

params[1]


for i in range(11):
    params = np.array(ss.iloc[i,1:5])
    print(bisect(integrate, -20, 2))
    


  
def LL(params,data): 
    # Params

    skew = params[0]
    kurt = params[1]
    # Standardize data
    
    
    # Compose PDF
    
    norm = 1 / (np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)
    temp1 = 1+skew/6 * (x**3 - 3*x) + (kurt-3)/24 * (x**4 - 6*x**2 + 3)
    temp2 = 1+skew**2 / 6 + (kurt-3)**2 / 24
    f = norm * temp1**2 / temp2
    
    # Log Maximum Likelihood Function
    ll = np.sum(np.log(f))
    
    return -ll


skew, kurt = minimize(LL, x0=np.array([-1,4]), args=spy, method = 'L-BFGS-B', bounds = ( (-200, 200), (0, 10)))['x']

skew, kurt = minimize(LL, x0=np.array([-1,4]), args=spy[:50], method = 'L-BFGS-B')['x']

def GC_Positive(x, skew, kurt):
    norm = 1 / (np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)
    temp1 = 1+skew/6 * (x**3 - 3*x) + (kurt-3)/24 * (x**4 - 6*x**2 + 3)
    temp2 = 1+skew**2 / 6 + (kurt-3)**2 / 24
    return norm *temp1**2 / temp2


x = np.linspace(-5, 5, 100000)
plt.figure(figsize = (15,10))
plt.plot(x,GC_Positive(x, skew, kurt))



area = 0
sets = np.linspace(-5, 5, 100000)
i = 0
while abs(area - 0.05) >= 0.001:
    a, b = GC_Positive(sets[i], skew, kurt), GC_Positive(sets[i+1], skew, kurt)
    area += (a+b)*(1/10000)
    i += 1
    
    
def GC_VaR(data):
    # Normalize Data
    x = (data - np.mean(data)) / np.std(data)
    # Log-Likelyhood Estimation
    skew, kurt = minimize(LL, x0 = np.array([-1,4]), args = x, method = 'L-BFGS-B')['x']
    # Compute VaR
    area = 0
    sets = np.linspace(-5, 5, 100000)
    i = 0
    while abs(area - 0.05) >= 0.001:
        a, b = GC_Positive(sets[i], skew, kurt), GC_Positive(sets[i+1], skew, kurt)
        area += (a+b)*(1/10000)
        i += 1
    
    return a

GC_SPY_50Day = spy.rolling(1900).apply(GC_VaR)

Log_Return = np.log(data/data.shift(1))


GC_SPY_100Day = pd.DataFrame()

for ETF in data.columns:
    GC_SPY_100Day[ETF] = 