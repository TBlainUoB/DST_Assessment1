# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:09:46 2022

@author: nd19620
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

#%%
train = pd.read_csv('Data/Newtrain.csv')

#%%
train_missing=dict()

for i in train.columns:
    train_missing.update({i:len(train[train[i] == -1])/train.shape[0]})
    
var_rand_miss = [key  for (key, value) in train_missing.items() if 0< value <0.05 ]

#%%
for i in var_rand_miss:
    trainData = train[train[i] != -1]

trainData.drop(['ps_car_03_cat','ps_car_05_cat'], inplace=True, axis=1)

#delete the variable with 45% and 70% missing rate 
#for the variables with missing rate less than 5%, delete rows with missing values for these random variables.

#%%
#Now we left with two variables 'ps_car_14' and 'ps_reg_03' to deal with, they have the corresponding missing rate 7% and 18%
mean1 = np.mean(train['ps_car_14'])
mean2 = np.mean(train['ps_reg_03'])
train['Im_ps_car_14'] = 0
train['Im_ps_reg_03'] = 0
for i in range(train.shape[0]):
    if train['ps_car_14'][i] == -1:
        train['Im_ps_car_14'][i] = mean1
    if train['ps_reg_03'][i] == -1:
        train['Im_ps_reg_03'][i] = mean2

                
                























