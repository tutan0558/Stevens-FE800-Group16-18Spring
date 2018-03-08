# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:24 2018

@author: 79127
"""

import numpy as np
import pandas as pd
import talib as ta
import tushare as ts
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# 确保可以显示‘-’号
mpl.rcParams['axes.unicode_minus']=False
# 确保中文显示正常
mpl.rcParams['font.sans-serif'] = ['SimHei']  

stock = ts.get_k_data('600030', '2016-06-01', '2017-06-30')
stock.sort_index(inplace=True)
stock.head()

stock['CCI'] = ta.CCI(np.array(stock.high), np.array(stock.low), np.array(stock.close), timeperiod  = 20)
stock.tail()


plt.subplot(2,1,1)
plt.title('600030 CCI Index')
plt.gca().axes.get_xaxis().set_visible(False)
stock['close'].plot(figsize = (10,8))
plt.legend()

plt.subplot(2,1,2)
stock.CCI.plot(figsize = (10,8))
plt.legend()


stock['yes_cci'] = stock.CCI.shift(1)
stock['daybeforeyes_cci'] = stock.CCI.shift(2)

stock['signal'] = np.where(np.logical_and(stock['daybeforeyes_cci'] < -100, stock['yes_cci'] > -100), 1, np.nan)
stock['signal'] = np.where(np.logical_and(stock['daybeforeyes_cci'] > 100, stock['yes_cci'] < 100), -1, stock['signal'])
stock['signal'] = stock['signal'].fillna(method = 'ffill')
stock = stock.fillna(0)


plt.subplot(3, 1, 1)
plt.title('600030 CCI开仓图')
plt.gca().axes.get_xaxis().set_visible(False)
stock['close'].plot(figsize = (10,8))
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
stock['CCI'].plot(figsize = (10,8))
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
stock['signal'].plot(figsize = (10,8),marker='o',linestyle='')

plt.legend(loc='upper left')
plt.show()


stock['pct_change'] = stock.close.pct_change()
stock['strategy_return'] = stock['pct_change'] * stock['signal']

stock['return'] = (stock['pct_change']+1).cumprod()
stock['strategy_cum_return'] = (1+stock['strategy_return']).cumprod()

stock[['return', 'strategy_cum_return']].plot(figsize = (10,6))

plt.title('600030 CCI收益图')
plt.legend()
plt.show()
