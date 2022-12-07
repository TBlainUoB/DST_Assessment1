#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:04:33 2022

@author: xin
"""
#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
#%%
trainData = pd.read_csv("Data/train.csv")
testData = pd.read_csv("Data/test.csv")

#%%
trainData.drop(['ps_car_03_cat','ps_car_05_cat'], inplace=True, axis=1)
testData.drop(['ps_car_03_cat','ps_car_05_cat'], inplace=True, axis=1)
#%%
col_to_drop = testData.columns[testData.columns.str.startswith('ps_calc_')]
trainData = trainData.drop(col_to_drop, axis=1)
testData = testData.drop(col_to_drop, axis=1)
#%%
def missingvalues(pdData):
    features = ['ps_reg_03', 'ps_car_12', 'ps_car_14', 'ps_car_11']
    pdData1 = pdData
    pdData1 = pdData1[pdData1['ps_car_14'] != -1]
    pdData1 = pdData1[pdData1['ps_reg_03'] != -1]
    pdData1 = pdData1[pdData1['ps_car_12'] != -1]
    pdData1 = pdData1[pdData1['ps_car_11'] != -1]
    X_train = pdData1.drop(['target', 'id','ps_car_14','ps_reg_03','ps_car_11','ps_car_12'], axis=1)
    X_train = pd.DataFrame(X_train)
    pdData0 = pdData.drop(['target', 'id','ps_car_14','ps_reg_03','ps_car_11','ps_car_12'], axis=1)
    for i in features:
            l_model = LinearRegression()
            y_train = pdData1[i].values
            l_model.fit(X_train,y_train)
            for j in range(pdData.shape[0]):
                if pdData[i].loc[j] == -1:
                    X = pdData0.loc[j]
                    X = pd.DataFrame(X).transpose()
                    pdData[i].loc[j] = l_model.predict(X)
    return pdData


#%%
trainData1= missingvalues(trainData)

