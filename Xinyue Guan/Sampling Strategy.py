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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  roc_curve,auc, accuracy_score
#%%
trainData = pd.read_csv('Data/New_train.csv')

#%%
train_size = math.floor(0.75*trainData.shape[0])
trainData = trainData.sample(frac = 1)
train = trainData.iloc[:train_size]
test = trainData.iloc[train_size:]

#%%
count_class_0, count_class_1 = train.target.value_counts()
df_class_0 = train[train['target'] == 0]
df_class_1 = train[train['target'] == 1]

#%%
df_class_0_under = df_class_0.sample(count_class_1)
train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
#combine the ones and the under sampled zeros, now we have a balanced dataset.

print('Random under-sampling:')
print(train_under.target.value_counts())

train_under.target.value_counts().plot(kind='bar', title='Count (target)')




#%%
y_train_under = train_under['target'].values
X_train_under = train_under.drop(['target', 'id'], axis=1)

X_test = test.drop(['target','id'], axis=1)
y_test = test['target'].values

X_train_under = pd.DataFrame(X_train_under)
X_test = pd.DataFrame(X_test)
y_train_under=y_train_under.ravel()

#%%
scaler = StandardScaler()
scaled_train0= scaler.fit_transform(X_train_under)
X_train_under = pd.DataFrame(scaled_train0, columns=X_train_under.columns)
scaled_test0 = scaler.fit_transform(X_test)
X_test = pd.DataFrame(scaled_test0, columns=X_test.columns)


#%%
knn_model1 = KNeighborsClassifier(n_neighbors=1)
knn_model1.fit(X_train_under, y_train_under)
fpr, tpr, threshold = roc_curve(y_test,knn_model1.predict(X_test))
print(auc(fpr, tpr))




#%%
knn_model3 = KNeighborsClassifier(n_neighbors=3)
knn_model3.fit(X_train_under, y_train_under)
fpr, tpr, threshold = roc_curve(y_test,knn_model3.predict(X_test))
print(auc(fpr, tpr))

#%%
knn_model5 = KNeighborsClassifier(n_neighbors=5)
knn_model5.fit(X_train_under, y_train_under)
fpr, tpr, threshold = roc_curve(y_test,knn_model5.predict(X_test))
print(auc(fpr, tpr))






















