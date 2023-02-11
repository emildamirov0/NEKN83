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
    model_stats = pd.DataFrame(index=['N_expeted', 'N_violations', 'Kupiec_p-value','Christoffersen_p-value' ])
    
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
        
        model_stats[model] = single_stats
        
        
        
    return df1, model_stats

viol_07, model_stats_07 = backtest(s_07)
viol_08, model_stats_08 = backtest(s_08) 




#%%

# def violations(sample):
    
#     models = ['BHS', 'EWMA', 'n','t', 'Pot']
#     df1 = sample['losses'].to_frame()
#     model_stats = {}
    
#     for model in models:
#         VaR_c = [col for col in sample.columns if model in col][0]
#         df2 = sample[VaR_c]
#         viol = sample['losses']>df2
#         viol = (viol.astype(int))
#         df1[model] = viol 
#         model_stats[model] = sum(viol)
 
#     return df1, model_stats

# viol_07, model_stats = violations(s_07)
# viol_08, model_stats = violations(s_08) 

