import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from numba import jit
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


def dropmissingcol(pdData):
    vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
    pdData.drop(vars_to_drop, inplace=True, axis=1)
    return pdData


def missingvalues(pdData):
    mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
    mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
    features = ['ps_reg_03', 'ps_car_12', 'ps_car_14', 'ps_car_11']
    for i in features:
        if i == 'ps_car_11':
            pdData[i] = mode_imp.fit_transform(pdData[[i]]).ravel()
        else:
            pdData[i] = mean_imp.fit_transform(pdData[[i]]).ravel()
    return pdData


def encodecat(train, test):
    cat_features = [col for col in train.columns if '_cat' in col]
    for column in cat_features:
        temp = pd.get_dummies(pd.Series(train[column]), prefix=column)
        train = pd.concat([train, temp], axis=1)
        train = train.drop([column], axis=1)

    for column in cat_features:
        temp = pd.get_dummies(pd.Series(test[column]), prefix=column)
        test = pd.concat([test, temp], axis=1)
        test = test.drop([column], axis=1)
    return train, test


def RescaleData(train, test):
    scaler = StandardScaler()
    scaler.fit_transform(train)
    scaler.fit_transform(test)
    return train, test


def DropCalcCol(train, test):
    col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(col_to_drop, axis=1)
    test = test.drop(col_to_drop, axis=1)
    return train, test


# read files from folder
train = pd.read_csv("Dataset/train.csv")
#test = pd.read_csv("Dataset/test.csv")

train, test = train_test_split(train, test_size=0.25, random_state=42)
train = pd.DataFrame(train)
test = pd.DataFrame(test)
train.to_csv("new_train.csv", index=False)
test.to_csv("new_test.csv", index=False)

train = dropmissingcol(train)
train = missingvalues(train)
test = dropmissingcol(test)
test = missingvalues(test)

y_train = train['target'].values
train_id = train['id'].values
X = train.drop(['target', 'id'], axis=1)

target_test = test['target'].values
test = test.drop(['target'], axis=1)
test_id = test['id']
X_test = test.drop(['id'], axis=1)

X, X_test = DropCalcCol(X, X_test)
X, X_test = encodecat(X, X_test)
X = pd.DataFrame(X)
X_test = pd.DataFrame(X_test)
X, X_test = RescaleData(X, X_test)

# StratifiedKFold
kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
pred_test_full = np.zeros(len(test_id))
cv_score = []

# Loops over each fold
for tr_id, te_id in kf.split(X, y_train):
    xtr, xvl = X.loc[tr_id], X.loc[te_id]
    ytr, yvl = y_train[tr_id], y_train[te_id]
    # initiates model
    lr = LogisticRegression(class_weight='balanced', C=0.003, max_iter=300)
    # fits model
    lr.fit(xtr, ytr)
    # predicts prob
    pred_test = lr.predict_proba(xvl)[:, 1]
    # evaluates with roc_auc_score

    score = gini(yvl, pred_test)
    print('gini', score)
    cv_score.append(score)
    # predict for test set
    pred_test_full += lr.predict_proba(X_test)[:, 1]
# average predictions over each fold
pred_test_full /= 5

pd.DataFrame({'id': test_id, 'target': pred_test_full}).to_csv('baseline_log_regression1.csv', index=False)
