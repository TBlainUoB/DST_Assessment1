import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

#read files from folder
train = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

#drop columns with large amount of missing variables
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
test.drop(vars_to_drop, inplace=True, axis=1)


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
missingvalues(train)
missingvalues(test)

#saves new datasets
train.to_csv('New_Dataset/train.csv')
test.to_csv('New_Dataset/test.csv')

#new vector of id
id_test = test['id'].values
id_train = train['id'].values
#target vector
y_train = train['target'].astype('category')
#removes these from the data as not relevant for model
del train['target']
del train['id']
del test['id']

#StratifiedKFold
kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = 0
cv_score = []
i = 1
#Loops over each fold
for id_train, id_test in kf.split(train, y_train):
    xtr,xvl = train.loc[id_train], train.loc[id_test]
    ytr, yvl = y_train[id_train], y_train[id_test]
    #initiates model
    lr = LogisticRegression(class_weight='balanced', C = 0.003)
    #fits model
    lr.fit(xtr, ytr)
    #predicts prob
    pred_test = lr.predict_proba(xvl)[:,1]
    #evaluates with roc_auc_score
    score = roc_auc_score(yvl, pred_test)
    print('roc_auc_score', score)
    cv_score.append(score)
    pred_test_full += lr.predict_proba(test)[:,1]
    i+=1

