#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:11:16 2022

@author: xin
"""
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #for standardizing data

#%%
trainData = pd.read_csv("/Users/xin/Desktop/porto-seguro-safe-driver-prediction/train.csv")

#%%
head = trainData.head()

#%%
trainData.shape
#We can see that there are 595,212 taining samples with 59 variables, one of the variables will be our target.

#%%
trainData.info()
#Info() tells us the data type of each variables

#%%
summary = trainData.describe()
#We can see that in our train data, there are no missing value for the target, which is the label we want to predict; 
#Also there are lot more 0's than 1's, because our mean is 0.03.

#%%
missing=dict()

for i in trainData.columns:
    missing.update({i:len(trainData[trainData[i] == -1])/trainData.shape[0]})
    
# we created a dictionary for each variable and its missing rate.

#%%
var_rand_miss = [key  for (key, value) in missing.items() if 0< value <0.05 ]
# a variable with missing rate less than 5%, we consider it as inconsequential, and therefore rows in these features could be dropped (Schafer 1999).

#%%
for i in var_rand_miss:
    trainData = trainData[trainData[i] != -1]
#For each of the variables with missing rate less than 5%, remove the rows with missing value.

trainData.drop(['ps_car_03_cat','ps_car_05_cat'], inplace=True, axis=1)
#We choose to drop the two variable which have the 69% and 45% missing rate.
#%%
missing1=dict()
missing1 = {k:v for (k,v) in missing.items() if 0.05<v<0.4}
#Now we left with two variables 'ps_car_14' and 'ps_reg_03' to deal with, they have the corresponding missing rate 7% and 18%.















#%%
#PCA is affected by scale, so we need to standardize our data before we apply PCA.
