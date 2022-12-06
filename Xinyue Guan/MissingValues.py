# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:09:46 2022

@author: nd19620
"""
#%%
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  roc_curve,auc, accuracy_score
#%%
train = pd.read_csv('Data/train.csv')

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
#Now we left with two variables 'ps_car_14' and 'ps_reg_03' to deal with, they have the corresponding missing rate 7% and 18%
col_to_drop = trainData.columns[trainData.columns.str.startswith('ps_calc_')]
trainData.drop(col_to_drop, inplace = True, axis=1)
#Similarly, drop the ps_calc_ variables for train and test data sets because these variables have no correlation with any other variables.

#%%
trainData = trainData[trainData['ps_car_14'] != -1]
trainData = trainData[trainData['ps_reg_03'] != -1]
#Remove all the missing value, we want create our random missing and test the performance of each imputation method.

#%%
# Split Data into Training and Testing in R 
train_size = math.floor(0.85*trainData.shape[0])
trainData = trainData.sample(frac = 1)
train = trainData.iloc[:train_size]
test = trainData.iloc[train_size:]
#randomly split dataset into 85% 15% train test data sets. training imputation algorithm in train data and then impute for test data set.



#%%                
error_mean_imp_14 = np.sum((np.mean(train['ps_car_14'])-test['ps_car_14'])**2)
error_mean_imp_03 = np.sum((np.mean(train['ps_reg_03'])-test['ps_reg_03'])**2)
#the mean square error calculated on test data set if the mean imputation is performed/

#%%
y_train14 = train['ps_car_14'].values
y_train14 = y_train14.ravel()
y_train03 = train['ps_reg_03'].values
X_train = train.drop(['target', 'id','ps_car_14','ps_reg_03'], axis=1)

X_test = test.drop(['target','id','ps_car_14','ps_reg_03'], axis=1)
y_test14 = test['ps_car_14'].values
y_test03 = test['ps_reg_03'].values

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
#%%
y_train = pd.DataFrame(y_train)
y_train1=y_train.to_numpy()
y_train1=y_train1.ravel()

#%%
knn_model3_14 = KNeighborsClassifier(n_neighbors=3)
knn_model3_14.fit(X_train,y_train14)
y_pred3_14 = knn_model3_14.predict(X_test)
print(accuracy_score(y_test14,y_pred3_14))
# 
fpr, tpr, threshold = roc_curve(y_test14, y_pred3_14)
print(auc(fpr, tpr))


















