#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:49:24 2022

@author: xin
"""
#%%
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
#%%
train = pd.read_csv("Data/imputetrain.csv")
test = pd.read_csv("Data/New_test.csv")


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

# store id and terget separately.

#%%
scaler = StandardScaler()
scaled_train0= scaler.fit_transform(X_train)
X_train = pd.DataFrame(scaled_train0, columns=X_train.columns)
scaled_test0 = scaler.fit_transform(X_test)
X_test = pd.DataFrame(scaled_test0, columns=X_test.columns)
# StandardScaler standardizes features by removing the mean and scaling to unit variance.


#%%
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = np.zeros(ceil(4*len(id_train)/5))
cv_score = []
knn_model1 = KNeighborsClassifier(n_neighbors=1)
knn_model3 = KNeighborsClassifier(n_neighbors=3)
knn_model5 = KNeighborsClassifier(n_neighbors=5)

#%%
scores1 = cross_val_score(knn_model1, X_train, y_train1, scoring='roc_auc',
                         cv=cv, n_jobs=-1)
    
#0.502806
#0.50558
#0.50661
#0.509524
#0.504529
#score with mean imputation

#0.504489
#0.508206
#0.506483
#0.505325
#0.503437
#score with regression impoutation
#%%
scores3 = cross_val_score(knn_model3, X_train, y_train1, scoring='roc_auc',
                         cv=cv, n_jobs=-1)
    

#0.509314
#0.509181
#0.514126
#0.513897
#0.513831

#0.517251
#0.515533
#0.513834
#0.511852
#0.512093



#%%
scores5 = cross_val_score(knn_model5, X_train, y_train1, scoring='roc_auc',
                         cv=cv, n_jobs=-1)


#0.517008
#0.514352
#0.520529
#0.516512
#0.518471

#0.525861
#0.522334
#0.519779
#0.516488
#0.513823

#By comparing the auc scores for 5-fold cross validation with mean imputation and regression imputation, we can see a very slight improvement
#by using regression imputation.








