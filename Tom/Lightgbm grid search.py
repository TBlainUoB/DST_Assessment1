import numpy as np
import pandas as pd
from datetime import datetime
from numba import jit
import lightgbm as lgbm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def timer(start_time=None):
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
    for i in range(n - 1, -1, -1):
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


Kaggle = False
impute = True
if Kaggle == False:
    if impute == True:
        train = pd.read_csv("imputeTrain.csv")
        test = pd.read_csv("imputetest.csv")
    else:
        train = pd.read_csv("new_train.csv")
        test = pd.read_csv("new_test.csv")
        test = dropmissingcol(test)
        train = dropmissingcol(train)
    target_test = test['target'].values
    test = test.drop(['target'], axis=1)
else:
    if impute == True:
        train = pd.read_csv("imputetrainKag.csv")
        test = pd.read_csv("imputetestKag.csv")
    else:
        train = pd.read_csv("Dataset/train.csv")
        test = pd.read_csv("Dataset/test.csv")
        test = dropmissingcol(test)
        train = dropmissingcol(train)

train = missingvalues(train)
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

OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50

params = {
    'min_child_weight': [5, 10, 12, 15, 30, 50, 100, 150],
    'num_leaves': [4, 5, 8, 10, 15, 20, 30],
    'subsample': [0.2, 0.4, 0.6, 0.8],
    'drop_rate': [0.1, 0.3, 0.5, 0.7, 0.15, 0.2],
    'max_depth': [3, 4, 5, 7, 10, 12, 15, 20]
}

model = lgbm.LGBMClassifier(learning_rate=LEARNING_RATE, n_estimators=600, objective='binary', )

folds = 5
param_comb = 40

SKfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4,
                                   cv=SKfold.split(X, y_train), verbose=3, random_state=1)

start_time = timer(None)

random_search.fit(X, y_train)
timer(start_time)

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best Normalised gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('lightgbm-randomgridsearch-results-03.csv')
