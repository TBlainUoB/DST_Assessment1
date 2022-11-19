import numpy as np
import pandas as pd
from datetime import datetime
from numba import jit

import lightgbm as lgbm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

def timer(start_time = None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds' % (thour, tmin, round(tsec, 2)))

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

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', gini(labels, preds), True

def dropmissingcol(pdData):
    vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
    pdData.drop(vars_to_drop, inplace=True, axis=1)
    return pdData

def missingvalues(pdData):
    mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
    mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
    features = ['ps_reg_03','ps_car_12','ps_car_14','ps_car_11']
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

train = pd.read_csv("Dataset/train.csv", dtype = {'id':np.int32, 'target':np.int8})
test = pd.read_csv("Dataset/test.csv", dtype={'id':np.int32})

train = dropmissingcol(train)
train = missingvalues(train)
test = dropmissingcol(test)
test = missingvalues(test)

y_train = train['target'].values
train_id = train['id'].values
X = train.drop(['target', 'id'], axis=1)

test_id = test['id']
X_test = test.drop(['id'], axis=1)

X, X_test = DropCalcCol(X, X_test)
X, X_test = encodecat(X, X_test)
X = pd.DataFrame(X)
X_test = pd.DataFrame(X_test)
X, X_test = RescaleData(X, X_test)

'''
Best hyperparameters from grid search:
{'subsample': 0.4, 'num_leaves': 5, 'min_child_weight': 15, 'max_depth': 3, 'drop_rate': 0.3}
'''

OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50

min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 10000
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": LEARNING_RATE,
          "num_leaves": 5,
          "max_bin": 256,
          "n_estimators": 600,
          "verbosity": 0,
          "feature_fraction": feature_fraction,
          "max_depth": 3,
          "drop_rate": 0.3,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 15,
          "min_split_gain": 0,
          "subsample": 0.4
          }

folds = 5

SKfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

best_trees = []
fold_scores = []

cv_train = np.zeros(len(y_train))
cv_pred = np.zeros(len(X_test))

start_time = timer(None)
iterations = 1
for seed in range(iterations):
    timer(start_time)
    params['seed'] = seed
    for id_train, id_test in SKfold.split(X, y_train):
        xtr, xvl = X.loc[id_train], X.loc[id_test]
        ytr, yvl = y_train[id_train], y_train[id_test]
        dtrain = lgbm.Dataset(data=xtr, label=ytr)
        dval = lgbm.Dataset(data=xvl, label=yvl, reference=dtrain)
        bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dval, feval=evalerror, verbose_eval=100,
                         early_stopping_rounds=100)

        best_trees.append(bst.best_iteration)
        fold_scores.append(bst.best_score)

        cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)

    cv_pred /= 5

pd.DataFrame({'id': test_id, 'target': cv_pred / iterations}).to_csv('lgbm_pred5-with-encodingscaling.csv', index=False)