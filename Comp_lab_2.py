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
def violations(sample):
    
    models = ['BHS', 'EWMA', 'n','t', 'Pot']
    df1 = sample['losses'].to_frame()
    n_viol = {}
    
    for model in models:
        VaR_c = [col for col in sample.columns if model in col][0]
        df2 = sample[VaR_c]
        viol = sample['losses']>df2
        viol = (viol.astype(int))
        df1[model] = viol 
        n_viol[model] = sum(viol)
 
    return df1, n_viol

viol_07, n_viol_07 = violations(s_07)
viol_08, n_viol_08 = violations(s_08) 

#%%    
# Kupiec
#for i in VaR_models:
 #   n_viol_07 = sum()
    
  # n_violations = sum(our_sample2007['violations_bhs2007']
   #n_expectd = 0.01*len(our_sample2007)
    
    
    
    #viol_07[str(i)] = viol
    
    
    
#s_07.insert(11,"Violations",np.nan)
#violations_bhs2007 = s_07['losses']>s_07['VaRBHS']
#s_07.insert(6,"v_BHS07",violations_bhs2007.astype(int))


