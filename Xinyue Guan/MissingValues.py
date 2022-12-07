# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:09:46 2022

@author: nd19620
"""
#%%
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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
# Split Data into Training and Testing datasets
train_size = math.floor(0.85*trainData.shape[0])
trainData = trainData.sample(frac = 1)
train = trainData.iloc[:train_size]
test = trainData.iloc[train_size:]
#randomly split dataset into 85% 15% train test data sets. training imputation algorithm in train data and then impute for test data set.



#%%                
error_mean_imp_14 = np.sum((np.mean(train['ps_car_14'])-test['ps_car_14'])**2)
#99.74392558930454
error_mean_imp_03 = np.sum((np.mean(train['ps_reg_03'])-test['ps_reg_03'])**2)
#6157.369879825044
#the mean square error calculated on test data set if the mean imputation is performed/

#%%
y_train14 = train['ps_car_14'].values
y_train14 = y_train14.ravel()
y_train03 = train['ps_reg_03'].values
X_train = train.drop(['target', 'id','ps_car_14','ps_reg_03'], axis=1)

X_test = test.drop(['target','id','ps_car_14','ps_reg_03'], axis=1)
y_test14 = test['ps_car_14'].values
y_test03 = test['ps_reg_03'].values

X_train0 = pd.DataFrame(X_train)
X_test0 = pd.DataFrame(X_test)

#%%
scaler = StandardScaler()
scaled_train0= scaler.fit_transform(X_train0)
X_train = pd.DataFrame(scaled_train0, columns=X_train.columns)
scaled_test0 = scaler.fit_transform(X_test0)
X_test = pd.DataFrame(scaled_test0, columns=X_test.columns)

#%%
knn_model3_14 = KNeighborsRegressor(n_neighbors=3)
knn_model3_14.fit(X_train,y_train14)
y_pred3_14 = knn_model3_14.predict(X_test)

error_kNN3_imp_14 = np.sum((y_pred3_14-test['ps_car_14'])**2)
#56.8354887070335


#%%
knn_model3_03 = KNeighborsRegressor(n_neighbors=3)
knn_model3_03.fit(X_train,y_train03)
y_pred3_03 = knn_model3_03.predict(X_test)

error_kNN3_imp_03 = np.sum((y_pred3_03-test['ps_reg_03'])**2)
#3513.5810431864893

#we can see that the kNN model with k=3 improves the performance of imputation quite significantly.



#%%
knn_model5_14 = KNeighborsRegressor(n_neighbors=5)
knn_model5_14.fit(X_train,y_train14)
y_pred5_14 = knn_model5_14.predict(X_test)

error_kNN5_imp_14 = np.sum((y_pred5_14-test['ps_car_14'])**2)
#51.939626752115494


#%%
knn_model5_03 = KNeighborsRegressor(n_neighbors=5)
knn_model5_03.fit(X_train,y_train03)
y_pred5_03 = knn_model5_03.predict(X_test)

error_kNN5_imp_03 = np.sum((y_pred5_03-test['ps_reg_03'])**2)
#3176.843399124529




#%%
l_model14 = LinearRegression()
l_model14.fit(X_train0,y_train14)
y_l_pred14 = l_model14.predict(X_test0)
error_multil_imp_14 = np.sum((y_l_pred14-test['ps_car_14'])**2)
#52.24618886814621


#%%
l_model03 = LinearRegression()
l_model03.fit(X_train0,y_train03)
y_l_pred03 = l_model03.predict(X_test0)
error_multil_imp_03 = np.sum((y_l_pred03-test['ps_reg_03'])**2)
#2473.9093364593655

#For linear regression, we need to use the dataset before standardizing

























