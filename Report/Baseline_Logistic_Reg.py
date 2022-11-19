import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

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


y_train = train['target'].values
id_train = train['id'].values
X = train.drop(['target', 'id'], axis=1)

id_test = test['id']
X_test = test.drop(['id'], axis=1)

col_to_drop = X.columns[X.columns.str.startswith('ps_calc_')]
X = X.drop(col_to_drop, axis=1)
X_test = X_test.drop(col_to_drop, axis=1)

cat_features = [col for col in X.columns if '_cat' in col]
for column in cat_features:
    temp = pd.get_dummies(pd.Series(X[column]), prefix=column)
    X = pd.concat([X, temp], axis=1)
    X = X.drop([column], axis=1)

for column in cat_features:
    temp = pd.get_dummies(pd.Series(X_test[column]), prefix=column)
    X_test = pd.concat([X_test, temp], axis=1)
    X_test = X_test.drop([column], axis=1)

X = pd.DataFrame(X)
X_test = pd.DataFrame(X_test)

scaler = StandardScaler()
scaler.fit_transform(X)
scaler.fit_transform(X_test)

#StratifiedKFold
kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = np.zeros(len(id_test))
cv_score = []

#Loops over each fold
for tr_id, te_id in kf.split(X, y_train):
    xtr,xvl = X.loc[tr_id], X.loc[te_id]
    ytr, yvl = y_train[tr_id], y_train[te_id]
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
    #predict for test set
    pred_test_full += lr.predict_proba(X_test)[:,1]
#average predictions over each fold
pred_test_full /= 5

pd.DataFrame({'id': id_test, 'target': pred_test_full}).to_csv('baseline_log_regression1.csv', index=False)
