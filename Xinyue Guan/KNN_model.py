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
from sklearn.metrics import  roc_curve,auc, accuracy_score
from math import ceil
from sklearn.ensemble import BaggingClassifier
#from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split,cross_val_score
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
knn_model1 = KNeighborsRegressor(n_neighbors=1) #default of distance metric is the Euclidean distance
# fits model
knn_model1.fit(X_train,y_train)
y_pred1 = knn_model1.predict(X_test)

#%%
n=len(y_pred1)
for i in range(0,n):
    if y_pred1[i]>0.5:
        y_pred1[i]=1
    else:
        y_pred1[i]=0

#%%
print(accuracy_score(y_test,y_pred1))
# 0.9348646645437575

#%%
fpr, tpr, threshold = roc_curve(y_test, y_pred1)
print(auc(fpr, tpr))
# 0.5035398151711877



#%%
knn_model3 = KNeighborsRegressor(n_neighbors=3) #default of distance metric is the Euclidean distance
# fits model
knn_model3.fit(X_train,y_train)
y_pred3 = knn_model3.predict(X_test)

#%%

n=len(y_pred3)
for i in range(0,n):
    if y_pred3[i]>0.5:
        y_pred3[i]=1
    else:y_pred3[i]=0
#%%
accuracy_score(y_test,y_pred3)
# running knn on training data once and then predict the test data set give an accuracy of 0.9610464477874163
fpr, tpr, threshold = roc_curve(y_test, y_pred3)
print(auc(fpr, tpr))
#auc score 0.5018147476426278
#%%

fpr, tpr, threshold = roc_curve(y_test, y_pred3)
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

#%%
knn_model5 = KNeighborsRegressor(n_neighbors=5) #default of distance metric is the Euclidean distance
# fits model
knn_model5.fit(X_train,y_train)
y_pred5 = knn_model5.predict(X_test)

#%%
n=len(y_pred5)
for i in range(0,n):
    if y_pred5[i]>0.5:
        y_pred5[i]=1
    else:
        y_pred5[i]=0

#%%
print(accuracy_score(y_test,y_pred5))
# 0.9638808059734271

#%%
fpr, tpr, threshold = roc_curve(y_test, y_pred5)
print(auc(fpr, tpr))
#0.5004445529097213

#%%
knn_model7 = KNeighborsRegressor(n_neighbors=7) #default of distance metric is the Euclidean distance
# fits model
knn_model7.fit(X_train,y_train)
y_pred7 = knn_model7.predict(X_test)

#%%
n=len(y_pred7)
for i in range(0,n):
    if y_pred7[i]>0.5:
        y_pred7[i]=1
    else:
        y_pred7[i]=0

#%%
print(accuracy_score(y_test,y_pred7))
# 0.963565114746898

#%%
fpr, tpr, threshold = roc_curve(y_test, y_pred7)
print(auc(fpr, tpr))
#0.5000693590760316
# increasing k results in a decrease of auc score, as we consider more neibours, there will be more zeros in our averaging calculation.

#%%
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = np.zeros(ceil(4*len(id_train)/5))
cv_score = []
knn_model = KNeighborsRegressor(n_neighbors=3)

#%%
scores = cross_val_score(knn_model, X_train, y_train, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)


#-0.0658454
#-0.0654451
#-0.0655671
#-0.0661047
#-0.0658492
# negative mean absolute value is not very helpful like the 'accuracy', due to the same reason our data is biased, with a lot more zeros
#%%
scores1 = cross_val_score(knn_model, X_train, y_train, scoring='roc_auc',
                         cv=cv, n_jobs=-1)
    

#0.504643
#0.511891
#0.511046
#0.505346



#%%
estimator_range = [2,4,6,8,10,12,14,16]

models = []
scores2 = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train)

    # Append the model and score to their respective list
    models.append(clf)
    scores2.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

#0.960561472859705
#0.9625013725705501
#0.9632608616082866
#0.9635811280699828
#0.9637549870063321
#0.9638739431206764
#0.9640020497053549
#0.9639745982943523

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
estimator_range = [10,12,14,16]
models2 = []
scores3 = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(base_estimator=knn_model1, n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train)

    # Append the model and score to their respective list
    models2.append(clf)
    
    y_pred = clf.predict(X_test)
    
    
    
#%%
    n=len(y_pred)
    for i in range(0,n):
        if y_pred[i]>0.5:
            y_pred[i]=1
        else:
            y_pred[i]=0

    fpr, tpr, threshold = roc_curve(y_test, y_pred)

    scores3.append(auc(fpr, tpr))



#%%

clf = BaggingClassifier(base_estimator=knn_model1, n_estimators = 4, random_state = 22)
clf.fit(X_train, y_train)


#%%
y_pred = clf.predict(X_test[1])


#%%
n=len(y_pred)
for i in range(0,n):
    if y_pred[i]>0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0

fpr, tpr, threshold = roc_curve(y_test, y_pred)




clf.predict(X_test.loc[17234])







