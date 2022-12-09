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
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #for standardizing data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  roc_curve,auc, accuracy_score
from numba import jit
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
#%%
@jit
def gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

#define the gini metric



#%%
trainData = pd.read_csv('Data/New_train.csv')

#%%
random.seed(42)
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
#0.5234220099241415
y_prob1 = knn_model1.predict_proba(X_test)
print(gini(y_test,y_prob1[:,1]))
#0.04668616773656997

#%%
knn_model3 = KNeighborsClassifier(n_neighbors=3)
knn_model3.fit(X_train_under, y_train_under)
fpr, tpr, threshold = roc_curve(y_test,knn_model3.predict(X_test))
print(auc(fpr, tpr))
#0.5375015397106558
y_prob3 = knn_model3.predict_proba(X_test)
print(gini(y_test,y_prob3[:,1]))
#0.10078438139595225


#%%
knn_model5 = KNeighborsClassifier(n_neighbors=5)
knn_model5.fit(X_train_under, y_train_under)
fpr, tpr, threshold = roc_curve(y_test,knn_model5.predict(X_test))
print(auc(fpr, tpr))
#0.5400291410844666
y_prob5 = knn_model5.predict_proba(X_test)
print(gini(y_test,y_prob5[:,1]))
#0.11375513330768572
#we see that under sampling does improved the performance of kNN, but not a lot, there not much improve by increasing k=3 to k=5, so for simple undersampling, k=3 should be chosen.
#Gini is approximately 2*roc-1
#%%
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
train_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(train_over.target.value_counts())

train_over.target.value_counts().plot(kind='bar', title='Count (target)');


#%%
y_train_over = train_over['target'].values
X_train_over = train_over.drop(['target', 'id'], axis=1)

X_test = test.drop(['target','id'], axis=1)
y_test = test['target'].values

X_train_over = pd.DataFrame(X_train_over)
X_test = pd.DataFrame(X_test)
y_train_over=y_train_over.ravel()


#%%
scaler = StandardScaler()
scaled_train0= scaler.fit_transform(X_train_over)
X_train_over = pd.DataFrame(scaled_train0, columns=X_train_over.columns)
scaled_test0 = scaler.fit_transform(X_test)
X_test = pd.DataFrame(scaled_test0, columns=X_test.columns)


#%%
knn_model1 = KNeighborsClassifier(n_neighbors=1)
knn_model1.fit(X_train_over, y_train_over)
fpr, tpr, threshold = roc_curve(y_test,knn_model1.predict(X_test))
print(auc(fpr, tpr))
#0.5064692862840756
y_prob1 = knn_model1.predict_proba(X_test)
print(gini(y_test,y_prob1[:,1]))
#0.02808695752667456

#%%
knn_model3 = KNeighborsClassifier(n_neighbors=3)
knn_model3.fit(X_train_over, y_train_over)
fpr, tpr, threshold = roc_curve(y_test,knn_model3.predict(X_test))
print(auc(fpr, tpr))
y_prob3 = knn_model3.predict_proba(X_test)
print(gini(y_test,y_prob3[:,1]))
#


#%%
knn_model5 = KNeighborsClassifier(n_neighbors=5)
knn_model5.fit(X_train_over, y_train_over)
fpr, tpr, threshold = roc_curve(y_test,knn_model5.predict(X_test))
print(auc(fpr, tpr))
y_prob5 = knn_model5.predict_proba(X_test)
print(gini(y_test,y_prob5[:,1]))
#

#%%
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


#%%
X = trainData.copy()
y_train = X['target'].values
X.drop(['id','target'],axis = 1)
pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, y_train, 'Imbalanced dataset (2 PCA components)')

#%%
rus = RandomUnderSampler(return_indices=True)
X_rus, y_rus, id_rus = rus.fit_sample(X, y_train)

print('Removed indexes:', id_rus)

plot_2d_space(X_rus, y_rus, 'Random under-sampling')


#There are a few more sampling strategy, for example, clustering before under/over sampling, so you reduce the same proportion for each cluster, however I didn't have time to try these.







