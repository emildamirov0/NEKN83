#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:22:58 2023

@author: emildamirov
"""

import pandas as pd
from scipy.stats import norm
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import math as m
import os
import datetime  as dt
import statistics as st


os.chdir('/Users/emildamirov/Downloads')
df = pd.read_excel('DataLab1.xlsx')
df["Date"] = pd.to_datetime(df["Date"])
df.set_index('Date', inplace=True)
print(df)


#%%

start = dt.datetime(2005, 1, 1)
end = dt.datetime(2008,12,31)
our_sample = df[start:end]

alpha = 0.99 # VaR/ES Level
our_sample['losses'] = our_sample['PL']*-1 # Don't forget the minus sign if we want the loss distribution
sorted_losses = our_sample.sort_values('losses', ascending=False)
print(sorted_losses)

T =  len(sorted_losses.index)# Get the number of returns
idx = m.floor((1-alpha)*T+1) # Calculate the index on the ordered data 
                           # that is our VaR (see formula on slide 6 of Non-parametric methods video)
print(sorted_losses.iloc[idx-1,1]) # we need iloc because we pick the row number not the index (which is a date)
                                   # -1 since first index is 0
print("On a X EUR portfolio we expect to lose less than {:.1f} USD 99% of the time".format(sorted_losses.iloc[idx-1,1]*1000))

es_99 = np.mean(sorted_losses.iloc[:idx-1,1]) # ES as mean of loss larger than VaR
print(sorted_losses.iloc[:idx-1,1]) # note that idx-1 actually excludes the VaR return
print("99% Expected Shortfall is {:.1f} USD".format(es_99*1000))

VaR_99_alt = np.quantile(our_sample['losses'],alpha)
# % Not exactly same since quantile uses linear interpolation
print("Percentage VaR {:.4f} %".format(VaR_99_alt*100))

es_99_alt = np.mean(our_sample.loc[our_sample['losses']>VaR_99_alt,'losses'])
print(es_99_alt)


#%%

# Task 1

rT = 500 # Size of rolling sample

our_sample.insert(2,"VaR",np.nan)

for j in range(rT,T): 
    roll_sample = our_sample.iloc[j-rT:j,1]
    our_sample.iloc[j,2] = np.quantile(roll_sample,0.99)
    

fig,ax=plt.subplots(nrows=1,ncols=1)
ax.plot(our_sample['VaR'])
plt.xlabel('Date',fontsize=12)
plt.ylabel('VaR (%)',fontsize=12)
plt.title('HS on 500 days rolling',fontsize=16)
ax.tick_params(axis='x',labelrotation=90,labelsize=9)
    

#%%

# Task 2a

s_mean = our_sample['losses'].iloc[0:500].mean()
our_sample['losses_new'] = our_sample['losses'] - s_mean
    

# Task 2b 
    
eta = our_sample.losses_new # Setting innovation equal to Loss is ok since daily returns close to zero and hard to predict
sigma = [np.var(eta.iloc[0:500])] # Sigma_0 in slides
lambda_par = 0.94 # RiskMetrics


for j in range(1,T):
    sigma.append((1-lambda_par)*eta[j-1]**2+lambda_par*sigma[j-1])

sigma = np.array(sigma)
annualized_percent_vol = (500*sigma)**0.5  # this is task 2c

fig,ax=plt.subplots()
ax.plot(annualized_percent_vol)
plt.title('Volatility from EWMA model',fontsize=14)

our_sample['std'] = sigma

# Task 2c

sigma_con = []

for i in sigma:
    sigma_con.append(m.sqrt(i))
    
our_sample['std_con'] = sigma_con

#%%
# Task 3

scaled_losses = np.zeros((rT,len(our_sample.losses_new)-rT))
scaled_losses.shape

for j in range(rT, T):
    scaled_losses[:,j-rT] = our_sample.std_con[j]/our_sample.std_con[j-rT:j]*our_sample.losses_new[j-rT:j]  
    
matrix = pd.DataFrame(scaled_losses)

#%%
# Task 4 

vw_VaR99 = []
vw_ES99 = [] 
our_sample.insert(6,"VW_VaR",np.nan)

for j in range(rT,T):
    vw_VaR99.append(np.quantile(matrix[j-rT],0.99))  
    interim = matrix[j-rT]
    vw_ES99.append(np.mean(interim[interim >vw_VaR99[j-rT]]))
    our_sample.iloc[j,6] = vw_VaR99[j-rT]    

    
fig,ax=plt.subplots()
ax.plot(our_sample['losses'][our_sample['losses']>0],label='Losses')
ax.plot(our_sample['VaR'],label='BHS')
ax.plot(our_sample['VW_VaR'],label='VWHS')

plt.xlabel('Date',fontsize=12)
plt.ylabel('VIX (%)',fontsize=12)
plt.title('BHS and VWHS on 500 days rolling',fontsize=16)
plt.legend()
fig.set_size_inches(12, 6)

#%%

# Task 5a
plt.hist(our_sample['losses'], edgecolor='black')

import numpy as np
import pylab
import scipy.stats as stats

# Quantile-Quantile plot:
    # If our data comes from a normal distribution, 
    # we should see all the points sitting on the straight line.

# Mean & Variance
print("Sample mean {:.2f}".format(st.mean(our_sample['losses'])))
print("Sample variance {:.2f}".format(st.variance(our_sample['losses'])))

# Histogram
import seaborn as sns
plot1 = sns.distplot(our_sample['losses'])
plt.hist(our_sample.losses, 30, density=True)

# Jarque-Berra
from scipy.stats import jarque_bera
data = our_sample['losses']
statistic,pvalue = jarque_bera(data)
print('Statistic=%.3f, P_value=%.3f\n' % (statistic, pvalue))

#Skewness
print("Sample skewness {:.2f}".format(stats.skew(our_sample.losses)))

#Kurtosis
print("Sample kurtosis {:.2f}".format(stats.kurtosis(our_sample.losses)))

# ('Losses have heavy tails and are right skewed, which will underestimate VaR and ES if we assume normality.')


# Task 5b
nu,loc,scale = stats.t.fit(our_sample.losses)
print('nu={:.2f} loc={:.4f}, and scale={:.4f}'.format(nu,loc,scale))

#our_sample.insert(7,"location",np.nan)
#our_sample.insert(8,"scale",np.nan)
#our_sample.insert(9,"degr_fr",np.nan)

#for j in range(rT,T): 
#    roll_sample = our_sample.iloc[j-rT:j,1]
#    our_sample.iloc[j,7] = stats.t.fit(roll_sample)[1]
#    our_sample.iloc[j,8] = stats.t.fit(roll_sample)[2]
#    our_sample.iloc[j,9] = stats.t.fit(roll_sample)[0]


Par_t = pd.DataFrame()
Par_t['losses'] = our_sample['losses']
Par_t.insert(1,"location",np.nan)
Par_t.insert(2,"scale",np.nan)
Par_t.insert(3,"degr_fr",np.nan)
Par_t['degr_fr'][500:1007] = 2.1

for j in range(rT,T): 
    roll_sample = our_sample.iloc[j-rT:j,1]
    Par_t.iloc[j,1] = stats.t.fit(roll_sample)[1]
    Par_t.iloc[j,2] = stats.t.fit(roll_sample)[2]

# Normal
Par_n = pd.DataFrame()
Par_n['losses'] = our_sample['losses']
Par_n.insert(1,"mean",np.nan)
Par_n.insert(2,"variance",np.nan)

for j in range(rT,T): 
    roll_sample = our_sample.iloc[j-rT:j,1]
    Par_n.iloc[j,1] = st.mean(roll_sample)
    Par_n.iloc[j,2] = st.variance(roll_sample)




    
