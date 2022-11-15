import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
test.drop(vars_to_drop, inplace=True, axis=1)



def missingvalues(data):
    mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
    mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
    features = ['ps_reg_03','ps_car_12','ps_car_14','ps_car_11']
    for i in features:
        if i == 'ps_car_11':
            data[i] = mode_imp.fit_transform(data[[i]]).ravel()
        else:
            data[i] = mean_imp.fit_transform(data[[i]]).ravel()

missingvalues(train)
missingvalues(test)


id_test = test['id'].values
id_train = train['id'].values
y_train = train['target'].astype('category')
del train['target']
del train['id']
del test['id']

kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = 0
cv_score = []
i = 1
print(len(train), len(y_train))
for id_train, id_test in kf.split(train, y_train):
    xtr,xvl = train.loc[id_train], train.loc[id_test]
    ytr, yvl = y_train[id_train], y_train[id_test]

    lr = LogisticRegression(class_weight='balanced', C = 0.003)
    lr.fit(xtr, ytr)
    pred_test = lr.predict_proba(xvl)[:,1]
    score = roc_auc_score(yvl, pred_test)
    print('roc_auc_score', score)
    cv_score.append(score)
    pred_test_full += lr.predict_proba(test)[:,1]
    i+=1
