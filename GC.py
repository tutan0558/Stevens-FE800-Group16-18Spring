# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:12:54 2018

@author: 79127
"""
import numpy as np
import statsmodels.sandbox.distributions.extras as extras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('seaborn')




GC = extras.pdf_mvsk([1, 2, 4, 2])
GC(0.05)
GC?


from scipy.stats import norm
norm.pdf(0.5)


norm.ppf?


x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100)
plt.plot(x, norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')


x = np.linspace(-6,6, 10000)
plt.plot(x, GC1(x = x, s = skew(spy), k = kurtosis(spy)), 'r-', lw = 5, alpha = 0.6)



x = np.linspace(-0.6,0.6, 10000)
plt.figure(figsize=(15, 5))
plt.title('Gram-Charlier Density')
plt.plot(x, GC2(x), 'r-', lw = 5, alpha = 0.6)
plt.axhline(0,color = 'black')





def GC1(x, s, k):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-(x**2)/2) * (1+(s/6)*(x**3 - 3*x) + k/24 * (x**4 - 6*x**2 + 3))


spy =np.log(data['SPY']/data['SPY'].shift(1))[:30].dropna()
GC2 = extras.pdf_mvsk([np.mean(spy), np.std(spy), skew(spy), kurtosis(spy)])


GC1(-2,s = skew(spy), k = kurtosis(spy))
GC2(-0.2)



def Posi(x):
    norm = (1/(np.sqrt(2*np.pi)))*np.exp(-x**2/2)
    temp1 = 1+(0.000002)/6*(x**3-3*x) + 3/24*(x**4-6*x**2+3)
    temp2 = 1+(0.000002)**2/6 + 3**2/24
    return norm*temp1**2/temp2

x = np.linspace(-100,100, 10000)

plt.plot(x, Posi(x), 'r-', lw = 5, alpha = 0.6)



def integrate(b):
    x = np.linspace(-10, b, 1000000)
    fx = Posi(x)
    area = np.sum(fx)*(b+10)/1000000
    return area -0.05

integrate(10)


bisect(integrate, -20,-2)
bisect()

Posi(2)
Posi(-2)
GC2(2)

(mean, sig, skew, kurt) = MLE_result


def Posi(x):
    norm = (1/(np.sqrt(2*np.pi*sig**2)))*np.exp(-(x-mean)**2/(2*sig**2))
    temp1 = 1+skew/6*(x**3-3*x) + (kurt-3)/24*(x**4-6*x**2+3)
    temp2 = 1+skew**2/6 + (kurt-3)**2/24
    return norm*temp1**2/temp2


mean, sig, skew, kurt = 0.005	,1,	-0.5628938	,3.75


def Posi2(x, skew, kurt):
    norm = (1/(np.sqrt(2*np.pi)))*np.exp(-(x)**2/(2*sig**2))
    temp1 = 1+skew/6*(x**3-3*x) + (kurt-3)/24*(x**4-6*x**2+3)
    temp2 = 1+skew**2/6 + (kurt-3)**2/24
    return norm*temp1**2/temp2


plt.plot(Posi2(x, -0.1, 6))

Posi(5) - Posi2(5)

GC30 = pd.read_csv('GC30.csv')
GC50 = pd.read_csv('GC50.csv')
GC30.drop(GC30.columns[0], axis = 1, inplace = True)
GC50.drop(GC50.columns[0], axis = 1, inplace = True)

GC50.hist(figsize = (15,8), bins = 30)
