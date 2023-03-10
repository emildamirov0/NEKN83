#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:25:30 2023

@author: emildamirov
"""

import pandas as pd
import datetime as dt
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from arch import arch_model 
pd.options.mode.chained_assignment = None 

df = pd.read_excel('DataLab2.xlsx')
df["Date"] = pd.to_datetime(df["Date"])
df.set_index('Date', inplace=True)

df['losses'] = df['PL']*-1
df.insert(6,'ESBHS',np.nan)
T =  len(df)
rT = 500 

for j in range(rT,T): 
    roll_sample = df.iloc[j-rT:j,11]
    sorted_losses = roll_sample.sort_values(ascending=False)
    df.iloc[j,6] = np.mean(sorted_losses[0:4])

start = dt.datetime(2007, 1, 1)
end = dt.datetime(2008,12,31)
our_sample = df[start:end]
our_sample['losses'] = our_sample['PL']*-1

s_07 = our_sample[dt.datetime(2007,1,1):dt.datetime(2007,12,31)]
s_08 = our_sample[dt.datetime(2008,1,1): dt.datetime(2008,12,31)]


#%%    

def backtest(sample):
    
    models = ['BHS', 'EWMA', 'n','t', 'Pot']
    df1 = sample['losses'].to_frame()
    model_stats = pd.DataFrame(index=['N_expeted', 'N_violations', 'Kupiec_p-value','Christoffersen_p-value', 'Z_2' ])
    
    for model in models:
        
        # Kupiec test
        single_stats = []
        VaR_c = [col for col in sample.columns if model in col][0]
        df2 = sample[VaR_c]
        viol = (sample['losses']>df2).astype(int)
        df1[model] = viol 
        single_stats.append(0.01*len(sample))
        single_stats.append(sum(viol))
        single_stats.append(1-stats.binom.cdf(sum(viol)-1,252,0.01))
        
        # Christoffersen test
        n1 = sum(viol)
        n0 = len(sample) - n1
        n11 = 0
        n00 = 0
        n01 = 0
        n10 = 0
        
        for r in range(0,len(sample)-1):
            if df1[model][r]==1 and df1[model][r+1]==1:
                n11 += 1
            if df1[model][r]==0 and df1[model][r+1]==0:
                n00 += 1
            if df1[model][r]==0 and df1[model][r+1]==1:
                n01 += 1
            if df1[model][r]==1 and df1[model][r+1]==0:    
                n10 += 1
        
        pi00 = n00 / (n00+n01) # Slide 8 VL 9
        pi01 = n01 / (n00+n01)
        pi10 = n01 / (n10+n11)
        pi11 = n11/ (n10+n11)
        pi0 = n0/ (n1+n0)
        pi1 = n1 / (n1+n0) 
        
        lnNull = np.log(pi0**n0*pi1**n1)
        lnAlt = np.log(pi00**n00*pi01**n01*pi10**n10*pi11**n11)
        LRind = -2*(lnNull-lnAlt)
        pVal = 1-stats.chi2.cdf(LRind,1)
        single_stats.append(pVal)
        
        # ES backtest
        M = len(sample) # Number of observations in backtesting sample
        alpha = 0.99
        ES_c = [col for col in sample.columns if model in col][1]
        ES = sample[ES_c]
        
        z2 = -M**-1*sum((df1[model]*sample['losses']/(1-alpha))/ES) + 1
        single_stats.append(z2)
        
        model_stats[model] = single_stats
        
    return df1, model_stats

#%%

viol_07, model_stats_07 = backtest(s_07)
viol_08, model_stats_08 = backtest(s_08) 


# To show particular results
model_stats_07.loc['Kupiec_p-value']
model_stats_08.loc['Kupiec_p-value']


# Christofferson independence test
# For each model in 2007, Christofferson test gives different results: BHS, n, t, POT have p-value over significance level 0,01; and EWMA has p-value less than 0,01. It indicates that EWMA is statistically rejected since violations are not independently distributed.
# For each model in 2008, Christofferson test gives different results: n, t, POT have p-value over 0,01; and BHS, EWMA have p-value less than 0,01, meaning that BHS and EWMA are rejected since violations are not independently distributed.
# In conclusion, EWMA does not pass Christofferson test in both 2007 and 2008, BHS does not pass the test for year 2008. Other models perform well.

# Traffic light test
# We don't need to make any additional calculations for the traffic light since the only thing that is needed is the number of violations. As for 2007, in BHS model we can see the number of violations are less than 5, which means they are in the green zone, so it indicates that the method that used to estimate ES does not have accuracy or quality problem. Then we can see the numbers of violations in EWMA and Pot models are equal to 5, so they are in amber zone and suggest the possibility that they have questions about accuracy and quality. And in n model and t model, the number of violations are much larger than 10, so they are in red zone and we can almost certainly say that there are problems with these ES methods. 
# And as for the data of 2008, the number of violations in Pot model is 3, which is less than 5 and in the green zone so we think it does not have accuracy and quality problem. The violation numbers in EWMA, BHS and t models are lager than 5 but smaller than 10, so they are in amber zone and show the probability of the problems with ES methods. But in n model the violation number is bigger than 10 and in the red zone, we can nearly ensure that this ES method has problems.
# In conclusion, for the 2007, it seems that the BHS method is the best. But for 2008, the Pot model has the highest quality and also the most accurate.


