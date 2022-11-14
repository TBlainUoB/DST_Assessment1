import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

MAX_ROUNDS = 400
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50

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
y = train['target']
del train['target']

model = XGBClassifier(
                        n_estimators=MAX_ROUNDS,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=LEARNING_RATE,
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                     )



