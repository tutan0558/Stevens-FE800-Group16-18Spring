# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:32:29 2018

@author: 79127
"""

import pandas as pd

import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import ConstantMean, GARCH, Normal

import statsmodels.stats as sts

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn')


data = pd.read_csv('800FinalData.csv',index_col = 0, parse_dates = True)
data.index = pd.to_datetime(data.index)

SPY = np.log(data.SPY / data.SPY.shift(1))

SPY.dropna(inplace = True)

md1 = smt.AR(SPY).fit(maxlag = 1, ic = 'aic', trend = 'nc')
eps = md1.resid
fittedvalue = md1.fittedvalues

md1.summary()
len(SPY)
from pandas import datetime
md1.predict(start = datetime(2018,2,1), end = datetime(2018,2,5))

predict1 = md1.predict(1, len(SPY))
md1.k_ar



res = am.fit()

res.summary()
