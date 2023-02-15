#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:13:52 2023

@author: emildamirov
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import os

os.chdir('/Users/emildamirov/Downloads')
risk_free = pd.read_excel('DataLab3rf.xlsx')
price_data = pd.read_excel('DataLab3prices.xlsx')
accounting_data = pd.read_excel('DataLab3accounting.xlsx')

#%%
def my_merton(x,sige,E,d,rf,t):
     d1 = (np.log(x[1]/d)+(rf+0.5*x[0]**2)*t)/(x[0]*np.sqrt(t))
     d2 = d1-x[0]*np.sqrt(t)
     f1 = x[1]*norm.cdf(d1)-np.exp(-rf*t)*d*norm.cdf(d2)-E # This is the equation on slide 21 Video L13 
     f2 = (x[1]/E)*norm.cdf(d1)*x[0]-sige # This is the equation on slide 22 Video L13 
     return np.sqrt(f1**2+f2**2) # Minimize f over x(1) and x(2) solves for Equity vol and Asset value

K_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


output = pd.DataFrame(index = K_list, columns = ['Asset_vol', 'Asset_val', 'DD', 'PD'])
output_1 = []


#%%
for i in K_list:
    equityReturns = np.log(price_data.Prices/price_data.Prices.shift(1))
    equityReturns = equityReturns[1:-1] # Removing the nan created by calculating returns
    sige = stats.tstd(equityReturns)*np.sqrt(250) # Standard deviation of equity (returns) scaled to yearly
    K = (accounting_data.LCT+i*accounting_data.DLTT) # We use 10% of long term debt which was found best in Afik et al. (2016)
    E = accounting_data.CSHO*price_data.Prices.iloc[-1] # Equity in MUSD as Number of shares * last share price
    rf = np.mean(risk_free.Riskfree) # Risk free rate as average of monthly rates
    t = 1 # We want PD in 1 year
    
    x0=(sige,K[0]+E[0]) # Initial quess for asset volatility and asset value,
                        # easier and faster for numerical optimization if guess is good
    x0bad = (sige,400)
    
    
    res = minimize(my_merton, x0, method='BFGS',
                   args=(sige,E[0],K[0],rf,t), options={'disp': True})

    
    DD = ((np.log(res.x[1])+(rf-0.5*res.x[0]**2)*np.sqrt(t)-np.log(K))/(res.x[0]*np.sqrt(t))) # DD slide 9 VL 13
    PD = float(norm.cdf(-DD))

    a = [res.x[0],res.x[1], DD, PD]
    output_1.append(a)
    
    output['Asset_vol'][i] = res.x[0]
    output['Asset_val'][i] = res.x[1]
    output['DD'][i] = DD
    output['PD'][i] = PD
    
     
#%%

fig,ax=plt.subplots(nrows=1,ncols=1)
ax.plot(output['PD'])
plt.xlabel('Date',fontsize=12)
plt.ylabel('K-value  (%)',fontsize=12)
plt.title('HS on 500 days rolling',fontsize=16)
ax.tick_params(axis='x',labelrotation=90,labelsize=9)
    

#%%
# Create dataset
# height = output['PD']
# bars = K_list
# x_pos = np.arange(len(bars))
 
# # Create bars
# plt.bar(x_pos, height)
 
# # Create names on the x-axis
# plt.xticks(x_pos, bars)
 
# # Show graphic
# plt.show()

#%%

k=0.5
K = accounting_data.LCT+k*accounting_data.DLTT 
E = accounting_data.CSHO*price_data.Prices.iloc[-1] 
rf = np.mean(risk_free.Riskfree) 
enddate=datetime(2015,12,31)
months=[1,3,6,9,12]
PD=[]
for j inrange(5):
    if j==0:
        startdate=datetime(2005,11,30) 
    elif j==1:
        startdate=datetime(2005,9,30)
    elif j==2:
        startdate=datetime(2005,6,30)
    elif j==3:
        startdate=datetime(2005,3,31)
    elif j==4:
        startdate=datetime(2004,12,31)




