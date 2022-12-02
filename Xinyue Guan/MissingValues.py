# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:09:46 2022

@author: nd19620
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

#%%#%%
trainData = pd.read_csv("Data/train.csv")

#%%
train_missing=dict()

for i in trainData.columns:
    train_missing.update({i:len(trainData[trainData[i] == -1])/trainData.shape[0]})
    
var_rand_miss = [key  for (key, value) in train_missing.items() if 0< value <0.05 ]

#%%
for i in var_rand_miss:
    trainData = trainData[trainData[i] != -1]

trainData.drop(['ps_car_03_cat','ps_car_05_cat'], inplace=True, axis=1)

#delete the variable with 45% and 70% missing rate 
#for the variables with missing rate less than 5%, delete rows with missing values for these random variables.

#%%
#Now we left with two variables 'ps_car_14' and 'ps_reg_03' to deal with, they have the corresponding missing rate 7% and 18%
imputer = SimpleImputer(strategy ='mean')
imputer.fit(trainData['ps_car_03_cat'])
trainData[Im_ps_car_03_cat]=imputer.transform(train_inputs[i])

#%%
imputer.fit(trainData['ps_car_05_cat'])
trainData[Im_ps_car_03_cat]=imputer.transform(train_inputs[i])






















