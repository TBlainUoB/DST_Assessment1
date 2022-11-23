#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:19:12 2022

@author: xin
"""
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #for standardizing data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,roc_auc_score, roc_curve, auc
from math import sqrt,ceil
#from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
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
# The test data set doesn't contain target, so we can't measure the performance of algorithms on it.
# So we can stratify k fold on our train data set, and each time leave one fold for measuring the performance of algorithm.
kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = np.zeros(ceil(4*len(id_train)/5))
cv_score = []

# Loops over each fold
for tr_id, te_id in kf.split(scaled_train, y_train):
    xtr, xvl = scaled_train.loc[tr_id], scaled_train.loc[te_id]
    ytr, yvl = y_train[tr_id], y_train[te_id]
    # initiates model
    knn_model = KNeighborsRegressor(n_neighbors=3) #default of distance metric is the Euclidean distance
    # fits model
    knn_model.fit(xtr, ytr)
    # predicts prob
    pred_test = knn_model.predict(xvl)[:, 1]
    # evaluates with roc_auc_score
    score = roc_auc_score(yvl, pred_test)
    print('roc_auc_score', score)
    cv_score.append(score)
    # predict for test set
    pred_test_full += knn_model.predict(xvl)[:, 1]
# average predictions over each fold
pred_test_full /= 5





#%%
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(scaled_train, y_train)
# fit data into our kNN model with k=3.

#%%
train_preds = knn_model.predict(scaled_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)


