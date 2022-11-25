#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:19:12 2022

@author: xin
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #for standardizing data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,roc_auc_score, roc_curve, auc, accuracy_score
from math import sqrt,ceil
#from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
#%%
train = pd.read_csv('Data/New_train.csv')
test = pd.read_csv('Data/New_test.csv')

#%%
y_train = train['target'].values
id_train = train['id'].values
X_train = train.drop(['target', 'id'], axis=1)

id_test = test['id']
X_test = test.drop(['id'], axis=1)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
# store id and terget separately.

#%%
scaler = StandardScaler()
scaled_train0= scaler.fit_transform(X_train)
scaled_train = pd.DataFrame(scaled_train0, columns=X_train.columns)
scaled_test0 = scaler.fit_transform(X_test)
scaled_test = pd.DataFrame(scaled_test0, columns=X_test.columns)
# StandardScaler standardizes features by removing the mean and scaling to unit variance.


#%%
#PCA is affected by scale, so we need to standardize our data before we apply PCA.


#%%
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,
                                   test_size=0.25, 
                                   shuffle=True)
#Because the test data set doesn't contain target, we need to split our train data into train and test data.

#%%
knn_model0 = KNeighborsRegressor(n_neighbors=3) #default of distance metric is the Euclidean distance
# fits model
knn_model0.fit(X_train,y_train)
y_pred = knn_model0.predict(X_test)

#%%
y_pred0=np.zeros(len(y_pred))
n=len(y_pred)
for i in range(0,n):
    if y_pred[i]>0.5:
        y_pred0[i]=1

#%%
accuracy_score(y_test,y_pred0)
# running knn on training data once and then predict the test data set give an accuracy of 0.9610464477874163


#%%

fpr, tpr, threshold = roc_curve(y_test, y_pred0)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

# The AUC score for our prediction is only approximately 0.5
# This is because there is a lot more zeros than ones in our dataset.

