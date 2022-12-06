# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:02:19 2022

@author: nd19620
"""
#In this file, we are aiming to deal with imbalanced data, check if this will improve the performance of kNN model

#%%
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #for standardizing data


#%%
trainData = pd.read_csv('Data/New_train.csv')

#%%
train_size = math.floor(0.75*trainData.shape[0])
trainData = trainData.sample(frac = 1)
train = trainData.iloc[:train_size]
test = trainData.iloc[train_size:]


#%%
y_train = train['target'].values
id_train = train['id'].values
X_train = train.drop(['target', 'id'], axis=1)

id_test = test['id'].values
X_test = test.drop(['target','id'], axis=1)
y_test = test['target'].values

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train1=y_train.ravel()

#%%
scaler = StandardScaler()
scaled_train0= scaler.fit_transform(X_train)
X_train = pd.DataFrame(scaled_train0, columns=X_train.columns)
scaled_test0 = scaler.fit_transform(X_test)
X_test = pd.DataFrame(scaled_test0, columns=X_test.columns)



#%%
count_class_0, count_class_1 = X_train.target.value_counts()
df_class_0 = X_train[X_train['target'] == 0]
df_class_1 = X_train[X_train['target'] == 1]

#%%
df_class_0_under = df_class_0.sample(count_class_1)
df_under = pd.concat([df_class_0_under, df_class_1], axis=0)
#combine the ones and the under sampled zeros, now we have a balanced dataset.

print('Random under-sampling:')
print(df_under.target.value_counts())

df_under.target.value_counts().plot(kind='bar', title='Count (target)')






















