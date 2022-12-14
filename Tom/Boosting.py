import gc

import numpy as np
import pandas as pd
from datetime import datetime

import xgboost
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
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




params = {
        'min_child_weight': [1,5,10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

MAX_ROUNDS = 400
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50

y_train = train['target'].values
X_train, X_test = train_test_split(train, test_size=0.75, stratify=y_train)

id_test = test['id'].values
id_train = X_train['id'].values
y_train = X_train['target'].values
X = X_train.drop(['target', 'id'], axis = 1)
Test_X = test.drop(['id'], axis=1)

model = XGBClassifier(
                        learning_rate = LEARNING_RATE, n_estimators = 600, objective='binary:logistic', nthread=1
                     )

folds = 3
param_comb = 5

SKfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4,
                                   cv= SKfold.split(X, y_train), verbose=3, random_state=1)

start_time = timer(None)

random_search.fit(X, y_train)
timer(start_time)
'''
model.fit(train,y_train)
preds = model.predict(test)
print(preds)
df = {'id': [test['id']], 'prediction': [preds]}
results = pd.DataFrame(df)
results.to_csv('Results.csv')
'''

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best Normalised gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-randomgridsearch-results-01.csv')