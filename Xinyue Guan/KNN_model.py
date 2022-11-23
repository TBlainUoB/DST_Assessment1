#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:19:12 2022

@author: xin
"""
#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler #for standardizing data
import numpy as np
#%%
train = pd.read_csv('Data/New_train.csv')
test = pd.read_csv('Data/New_test.csv')

#%%
scaler = StandardScaler()