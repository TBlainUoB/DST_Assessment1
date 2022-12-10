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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  roc_curve,auc, accuracy_score, confusion_matrix
from math import ceil
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split,cross_val_score,cross_val_predict
import time
#%%
train = pd.read_csv('Data/New_train.csv')
test = pd.read_csv('Data/New_test.csv')

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
pca = PCA(svd_solver='full')
pca.fit(X_train)
#PCA is affected by scale, so we need to standardize our data before we apply PCA.
#%%
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
feat_labels = X_train.columns
EVR = pca.explained_variance_ratio_
indices = np.argsort(pca.explained_variance_ratio_)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], EVR[indices[f]]))

SingularV = pca.singular_values_
indices = np.argsort(pca.singular_values_)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], SingularV[indices[f]]))

#%%
#Another method for feature selection
feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train1)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))





#%%
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
knn_model1 = KNeighborsClassifier(n_neighbors=1)
knn_model3 = KNeighborsClassifier(n_neighbors=3)
knn_model5 = KNeighborsClassifier(n_neighbors=5)



#%%

scores1 = cross_val_score(knn_model1, X_train, y_train1, scoring='roc_auc', cv=cv, n_jobs=-1)
scores1.to_csv('1NN_5-fold_cv_auc.csv')



#%%
scores1 = cross_val_score(knn_model1, X_train, y_train1, scoring='roc_auc',
                         cv=cv, n_jobs=-1)
    
#0.502806
#0.50558
#0.50661
#0.509524
#0.504529

#This is code to implemetn k-fold cross validation.

#%%

scores = [[0.504489,0.508206,0.506483,0.505325,0.503437],
          [0.517251,0.515533,0.513834,0.511852,0.512093],
          [0.525861,0.522334,0.519779,0.516488,0.513823]]
scores = pd.DataFrame(scores,columns=['1st_fold', '2nd_fold', '3rd_fold','4th_fold','5th_fold'])
scores.to_csv('C:/Users/nd19620/OneDrive - University of Bristol/DST/DST_Assessment1/Report/auc_full_training_set.csv', index = False)

#%%
scores3 = cross_val_score(knn_model3, X_train, y_train1, scoring='roc_auc',
                         cv=cv, n_jobs=-1)
    

#0.509314
#0.509181
#0.514126
#0.513897
#0.513831




#%%
scores5 = cross_val_score(knn_model5, X_train, y_train1, scoring='roc_auc',
                         cv=cv, n_jobs=-1)


#0.517008
#0.514352
#0.520529
#0.516512
#0.518471


#%%
data = [[0.502806,0.50558,0.50661,0.509524,0.504529],[0.509314,0.509181,0.514126,0.513897,0.513831],[0.517008,0.514352,0.520529,0.516512,0.518471]]

for k in range(3):
    for i in range(5):
        plot(data[k,i],shape=)



#%%
estimator_range = [2,4,6,8,10,12,14,16]

models = []
scores2 = []
scores3 = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train1)

    # Append the model and score to their respective list
    models.append(clf)
    
    fpr, tpr, threshold = roc_curve(y_test,clf.predict(X_test))
    scores2.append(auc(fpr, tpr))
    
    scores3.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

#0.5007266093314129
#0.5004023546893056
#0.5002843206803768
#0.5002615962594202
#0.49993122886859737
#0.4999775281327952
#0.4999917740602407
#0.500122445913075


#0.9601719178035166
#0.962368950436317
#0.9631997034005946
#0.9635086611145821
#0.9635773183843571
#0.9636665728350646
#0.9636940357429746
#0.9637695587397271

#The default parameter for the base classifier in BaggingClassifier is the DicisionTreeClassifier
#%%
# Generate the plot of scores against number of estimators
plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores2)
plt.title ('Bagging with default baseline estimator')
# Adjust labels and font (to make visable)
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

# Visualize plot
plt.show()



#%%
knn_model1 = KNeighborsClassifier(n_neighbors=1)
clf = BaggingClassifier(base_estimator=knn_model1, n_estimators = 4, random_state = 22)
y_train=np.ravel(y_train)
clf.fit(X_train, y_train)
fpr, tpr, threshold = roc_curve(y_test,clf.predict(X_test))
print(auc(fpr, tpr))


#0.5052903984913617


#%%
knn_model1 = KNeighborsClassifier(n_neighbors=1)
clf = BaggingClassifier(base_estimator=knn_model1, n_estimators = 8, random_state = 22)
y_train=np.ravel(y_train)
clf.fit(X_train, y_train)
fpr, tpr, threshold = roc_curve(y_test,clf.predict(X_test))
print(auc(fpr, tpr))
# 0.5047075782348553

#Bagging only make sense with k=1 because otherwise kNN is too stable, but it doesn't improve the performance of the kNN model much.

