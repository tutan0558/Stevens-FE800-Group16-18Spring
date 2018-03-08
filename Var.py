# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:28:50 2018

@author: 79127
"""

import numpy as np
from scipy.optimize import bisect

from numba import jit
S = 1
K = 3
def f(x, s, k):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-(x**2)/2) * (1+(s/6)*(x**3 - 3*x) + (k-3)/24 * (x**4 - 6*x**2 + 3))



def f(x):
    return (((1/np.sqrt(2*np.pi))*np.exp(x**2/2))*(1+(-0.9823/6)*(x**3-3*x)+((0.1461011)/24)*(x**4-6*x**2+3))**2)/(1+0.9823**2/6+(0.1461011)**2/24)




(((1/sqrt(2*pi))*exp(x^2/2))*(1+(s/6)*(x^3-3*x)+((k-3)/24)*(x^4-6*x^2+3))^2)/(1+s^2/6+(k-3)^2/24)


@jit()
def integrate(b):
    x = np.linspace(-200, b, 1000000)
    fx = f(x)
    area = np.sum(fx)*(b+200)/1000000
    return area -0.95

 
integrate(200)   


integrate(1000)

bisect(integrate, -5, 5)





