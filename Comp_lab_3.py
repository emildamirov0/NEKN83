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
from datetime import datetime
import os
import plotly.express as px
from matplotlib.figure import Figure

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
# height = output['PD']
# bars = K_list
# x_pos = np.arange(len(bars))
# plt.bar(x_pos, height)
# plt.xticks(x_pos, bars)
# plt.show()

#%%

# Task 2
# Volatility estimation

price_data["Date"] = pd.to_datetime(price_data["Date"], format = '%Y%m%d')
price_data_tt = price_data.set_index('Date')

k=0.5
K = accounting_data.LCT+k * accounting_data.DLTT
E = accounting_data.CSHO*price_data.Prices.iloc[-1] 
rf = np.mean(risk_free.Riskfree) 
t = 1
enddate=datetime(2015,12,31)
months=[1,3,6,9,12]
sige = np.zeros(5)
PD_v=[]

#%%
for j in range(5):
    
    if j==0:
        startdate=datetime(2015,11,30) 
    elif j==1:
        startdate=datetime(2015,9,30)
    elif j==2:
        startdate=datetime(2015,6,30)
    elif j==3:
        startdate=datetime(2015,3,31)
    elif j==4:
        startdate=datetime(2014,12,31)

    TR = (price_data_tt.index >= startdate) & (price_data_tt.index <= enddate)
    equityReturns = price_data_tt.loc[TR, 'Prices'].pct_change().dropna()
    
    sige[j] = equityReturns.std() * np.sqrt(250)
    res = minimize(my_merton, x0, method = 'BFGS',
                  args = (sige[j], E, K, rf, t), options = {'disp': False})
    
    #DD = (np.log(res.x[1])+(rf-0.5*res.x[0]**2)*t-np.log(K))/(res.x[0]*np.sqrt(t)) # DD slide 9 VL 13
    DD = (np.log(res.x[1])+(rf-0.5*sige[j]**2)*t-np.log(K))/(sige[j]*np.sqrt(t)) # DD slide 9 VL 13
    
    
    PD = float(norm.cdf(-DD))
    PD_v.append(PD)

#%%
X = ['One month','Three months', 'Six months', 'Nine months', 'One year' ]
plt.bar(X, PD_v)
plt.ylabel('Equity volatility')
fig.set_size_inches(10,6)


#%%

# Task 3: VWHS

eta = equityReturns # Setting innovation equal to Loss is ok since daily returns close to zero and hard to predict
T = len(eta)
sigma = np.zeros(T+1)# Sigma_0 in slides
Lambda = 0.94 # RiskMetrics
sigma[0] = np.var(equityReturns)

for j in range(1,T+1):
    
    sigma[j] = ((1-Lambda)*(eta.iloc[j-1]-np.mean(eta))**2+ Lambda * sigma[j-1])


sigel = np.mean((np.sqrt (sigma)) * np.sqrt (250)) # Sigma is a variance
res = minimize (my_merton, x0, method='BFGS',
                args=(sigel,E,K,rf,t), options={'disp': False})

DDEWMA = (np.log(res.x[1])+(rf-0.5*sigel**2) *t-np.log(K))/(sigel*np.sqrt(t)) # DD slide 9 VL 13

PDEWMA = norm.cdf(-DDEWMA)
PD_v.append(PDEWMA.item())
sige = np.append (sige , sigel)


#%%
fig,ax=plt.subplots(nrows=1,ncols=1)

X = ['One month', 'three months','six months', 'nine months', '1 year', 'EWMA']
plt.bar(X, PD_v)
plt.ylabel('PD')
fig.set_size_inches(10,6)
