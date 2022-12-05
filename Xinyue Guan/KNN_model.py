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
from sklearn.model_selection import StratifiedKFold, train_test_split,cross_val_score
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
y_train = pd.DataFrame(y_train)
y_train1=y_train.to_numpy()
y_train1=y_train1.ravel()

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
knn_model1 = KNeighborsClassifier(n_neighbors=1) #default of distance metric is the Euclidean distance
# fits model
knn_model1.fit(X_train,y_train1)
y_pred1 = knn_model1.predict(X_test)


#%%
print(accuracy_score(y_test,y_pred1))
#0.9330522962423876

#%%
fpr, tpr, threshold = roc_curve(y_test, y_pred1)
print(auc(fpr, tpr))
#auc score 0.5035854331654345



#%%
knn_model3 = KNeighborsClassifier(n_neighbors=3) #default of distance metric is the Euclidean distance
# fits model
knn_model3.fit(X_train,y_train1)
y_pred3 = knn_model3.predict(X_test)

#%%
print(accuracy_score(y_test,y_pred3))
# running knn on training data once and then predict the test data set give an accuracy of 0.9604877412444817
fpr, tpr, threshold = roc_curve(y_test, y_pred3)
print(auc(fpr, tpr))
#auc score 0.5021713752300643
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
knn_model5 = KNeighborsClassifier(n_neighbors=5) #default of distance metric is the Euclidean distance
# fits model
knn_model5.fit(X_train,y_train1)
y_pred5 = knn_model5.predict(X_test)

#%%
print(accuracy_score(y_test,y_pred5))
#0.9635429897494696

#%%
fpr, tpr, threshold = roc_curve(y_test, y_pred5)
print(auc(fpr, tpr))
#0.5006453858781639

# increasing k results in a decrease of auc score, as we consider more neibours, there will be more zeros in our averaging calculation.

#%%
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = np.zeros(ceil(4*len(id_train)/5))
cv_score = []
knn_model = KNeighborsClassifier(n_neighbors=3)

#%%
scores = cross_val_score(knn_model, X_train, y_train1, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)


#-0.0386551
#-0.0386441
#-0.0387928
#-0.0385412
#-0.0386326

#This is code to implemetn k-fold cross validation.
# negative mean absolute value is not very helpful like the 'accuracy', due to the same reason our data is biased, with a lot more zeros
#%%
scores1 = cross_val_score(knn_model, X_train, y_train1, scoring='roc_auc',
                         cv=cv, n_jobs=-1)
    

#0.509314
#0.509181
#0.514126
#0.513897
#0.513831




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
estimator_range = [2,4,6,8,10,12,14,16]

models2 = []
scores4 = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train1)

    # Append the model and score to their respective list
    models2.append(clf)
    y_pred = clf.predict(X_test)
    fpr, tpr, threshold = roc_curve(y_test,y_pred)
    scores3.append(auc(fpr, tpr))

#0.501898851810379
#0.5000556758805177
#0.5003469887545138
#0.5003145043042855
#0.5002179720316834
#0.5001572521521428
#0.5000716247861234
#0.5000858576355398

#%%
#knn_model1 = KNeighborsClassifier(n_neighbors=1)
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
knn_model1 = KNeighborsClassifier(n_neighbors=1)
clf = BaggingClassifier(base_estimator=knn_model1, n_estimators = 4, random_state = 22)
y_train=np.ravel(y_train)
clf.fit(X_train, y_train)


#%%
y_pred = clf.predict(X_test)

#This code doesn't end somehow






