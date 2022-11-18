import numpy as np
import pandas as pd
from datetime import datetime

import lightgbm as lgbm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

def timer(start_time = None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds' % (thour, tmin, round(tsec, 2)))

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True


train = pd.read_csv("Dataset/train.csv", dtype = {'id':np.int32, 'target':np.int8})
test = pd.read_csv("Dataset/test.csv", dtype={'id':np.int32})

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

y_train = train['target'].values
train_id = train['id'].values
X = train.drop(['target', 'id'], axis=1)

test_id = test['id']
X_test = test.drop(['id'], axis=1)

'''
Best hyperparameters:
{'subsample': 0.4, 'num_leaves': 5, 'min_child_weight': 15, 'max_depth': 3, 'drop_rate': 0.3}
'''

OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50


'''
model = lgbm.LGBMClassifier(
    subsample = 0.4, num_leaves=5, min_child_weight=15, max_depth=3, drop_rate=0.3,
    learning_rate = LEARNING_RATE, n_estimators = 600, objective='binary')
'''
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
print(len(cv_pred))

for seed in range(20):
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

pd.DataFrame({'id': test_id, 'target': cv_pred / 20}).to_csv('lgbm_pred20.csv', index=False)