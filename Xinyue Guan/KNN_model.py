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
from sklearn.metrics import  roc_curve,auc, accuracy_score
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
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,
                                   test_size=0.25, 
                                   shuffle=True)
#Because the test data set doesn't contain target, we need to split our train data into train and test data.


#%%
pca = PCA(svd_solver='full')
pca.fit(X_train)
#PCA is affected by scale, so we need to standardize our data before we apply PCA.
#%%
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

#%%
feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))

 #1) ps_car_13                      0.122827
 #2) ps_reg_03                      0.113170
 #3) ps_car_14                      0.075220
 #4) ps_ind_15                      0.066739
 #5) ps_ind_03                      0.063638
 #6) ps_reg_02                      0.059400
 #7) ps_car_11_cat                  0.052956
 #8) ps_car_15                      0.049726
 #9) ps_ind_01                      0.049659
#10) ps_reg_01                      0.048482
#11) ps_car_01_cat                  0.041342
#12) ps_car_06_cat                  0.038869
#13) ps_car_12                      0.035021
#14) ps_car_09_cat                  0.024025
#15) ps_ind_02_cat                  0.021459
#16) ps_car_11                      0.018941
#17) ps_ind_04_cat                  0.017020
#18) ps_ind_05_cat                  0.013306
#19) ps_ind_16_bin                  0.009065
#20) ps_car_04_cat                  0.008925
#21) ps_car_08_cat                  0.008196
#22) ps_ind_08_bin                  0.007900
#23) ps_car_02_cat                  0.007690
#24) ps_ind_18_bin                  0.007552
#25) ps_ind_07_bin                  0.007463
#26) ps_ind_09_bin                  0.007367
#27) ps_ind_06_bin                  0.006812
#28) ps_car_07_cat                  0.006471
#29) ps_ind_17_bin                  0.004591
#30) ps_car_10_cat                  0.002101
#31) ps_ind_14                      0.001816
#32) ps_ind_12_bin                  0.001535
#33) ps_ind_11_bin                  0.000373
#34) ps_ind_13_bin                  0.000269
#35) ps_ind_10_bin                  0.000076
#%%
knn_model1 = KNeighborsClassifier(n_neighbors=1) #default of distance metric is the Euclidean distance
# fits model
knn_model1.fit(X_train,y_train)
y_pred1 = knn_model1.predict(X_test)


#%%
print(accuracy_score(y_test,y_pred1))
# 0.9348646645437575

#%%
fpr, tpr, threshold = roc_curve(y_test, y_pred1)
print(auc(fpr, tpr))
# 0.5035398151711877



#%%
knn_model3 = KNeighborsClassifier(n_neighbors=3) #default of distance metric is the Euclidean distance
# fits model
knn_model3.fit(X_train,y_train)
y_pred3 = knn_model3.predict(X_test)

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
knn_model5 = KNeighborsClassifier(n_neighbors=5) #default of distance metric is the Euclidean distance
# fits model
knn_model5.fit(X_train,y_train)
y_pred5 = knn_model5.predict(X_test)

#%%
print(accuracy_score(y_test,y_pred5))
# 0.9638808059734271

#%%
fpr, tpr, threshold = roc_curve(y_test, y_pred5)
print(auc(fpr, tpr))
#0.5004445529097213

#%%
knn_model7 = KNeighborsClassifier(n_neighbors=7) #default of distance metric is the Euclidean distance
# fits model
knn_model7.fit(X_train,y_train)
y_pred7 = knn_model7.predict(X_test)


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
knn_model = KNeighborsClassifier(n_neighbors=3)

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
estimator_range = [2,4,6,8,10,12,14,16]

models2 = []
scores3 = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train)

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






