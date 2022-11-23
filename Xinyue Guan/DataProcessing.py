#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:11:16 2022

@author: xin
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

#%%
trainData = pd.read_csv("/Users/xin/Library/CloudStorage/OneDrive-UniversityofBristol/DST/DST_Assessment1/Xinyue Guan/Data/train.csv")
testData = pd.read_csv("/Users/xin/Library/CloudStorage/OneDrive-UniversityofBristol/DST/DST_Assessment1/Xinyue Guan/Data/test.csv")

#%%
head = trainData.head()

#%%
trainData.shape
#We can see that there are 595,212 taining samples with 59 variables, one of the variables will be our target.

#%%
trainData.info()
#Info() tells us the data type of each variables

#%%
summary = trainData.describe()
#We can see that in our train data, there are no missing value for the target, which is the label we want to predict; 
#Also there are lot more 0's than 1's, because our mean is 0.03.

#%%
train_missing=dict()

for i in trainData.columns:
    train_missing.update({i:len(trainData[trainData[i] == -1])/trainData.shape[0]})
    
# we created a dictionary for each variable and its missing rate for the train data.

#%%
test_missing=dict()

for i in testData.columns:
    test_missing.update({i:len(testData[testData[i] == -1])/testData.shape[0]})
    
# we created a dictionary for each variable and its missing rate for the test data.

#%%
var_rand_miss1 = [key  for (key, value) in train_missing.items() if 0< value <0.05 ]
var_rand_miss2 = [key  for (key, value) in test_missing.items() if 0< value <0.05 ]
# find the variables that have missing rate less than 5% for train and test data, and it turns out they are the same set of variables
# a variable with missing rate less than 5%, we consider it as inconsequential, and therefore rows in these features could be dropped (Schafer 1999).
# therefore we can just delete these variables in both data sets.
#%%
for i in var_rand_miss1:
    trainData = trainData[trainData[i] != -1]
    testData = testData[testData[i] != -1]
#For each of the variables with missing rate less than 5%, remove the rows with missing value.

trainData.drop(['ps_car_03_cat','ps_car_05_cat'], inplace=True, axis=1)
testData.drop(['ps_car_03_cat','ps_car_05_cat'], inplace=True, axis=1)
#We choose to drop the two variable which have the 70% and 45% missing rate in both train and test data sets.
#%%
missing1=dict()
missing1 = {k:v for (k,v) in train_missing.items() if 0.05<v<0.4}

missing2=dict()
missing2 = {k:v for (k,v) in test_missing.items() if 0.05<v<0.4}

#Now we left with two variables 'ps_car_14' and 'ps_reg_03' to deal with, they have the corresponding missing rate 7% and 18% in both train and test data sets.
#%%
#Let's firstly try the simple imputation methods to see how the models work on the imputed dataset.
#function to replace missing values in data
def missingvalues(data):
    mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
    mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
    features = ['ps_reg_03','ps_car_12','ps_car_14','ps_car_11']
    for i in features:
        if i == 'ps_car_11':
            data[i] = mode_imp.fit_transform(data[[i]]).ravel()
        else:
            data[i] = mean_imp.fit_transform(data[[i]]).ravel()

#replace missing values on both train and test sets
missingvalues(trainData)
missingvalues(testData)

#%%
trainData['target'].value_counts().plot(kind='bar', figsize=(5,5))
# we can see that there is a lot more of 0s than 1s, as expected. The dataset is not balanced.

#%%
fig, ax = plt.subplots(figsize=(20,10))         
corr = trainData.corr()
sns.heatmap(corr, cmap='RdYlBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14)
plt.show()
# corrrelation plot, we can see that there is no correlation between target and the calc variables.

#%%
col_to_drop = testData.columns[testData.columns.str.startswith('ps_calc_')]
trainData = trainData.drop(col_to_drop, axis=1)
testData = testData.drop(col_to_drop, axis=1)
#Drop the ps_calc_ variables for train and test data sets.

#%%
# saves new datasets
trainData.to_csv('Data/New_train.csv',index=False)
testData.to_csv('Data/New_test.csv',index=False)
# these data sets are now ready for implementing algorithms.




#%%
#PCA is affected by scale, so we need to standardize our data before we apply PCA.
