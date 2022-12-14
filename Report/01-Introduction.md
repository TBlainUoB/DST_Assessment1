# Introduction

Inaccuracies in car insurance companies claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones. If insurance companies can predict the claim rate of drivers more accurately, it will allow them to tailor their prices further. And hence, they will not lose profit by setting the price too low and lose customers by setting the price too high. In this project, we use the dataset from Porto Seguro, one of Brazils largest auto and homeowner insurance companies, and try to build models that predict the probability that a driver will initiate an auto insurance claim in the next year. 

Our project is based on the following Kaggle competition https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction. 

We have used the train dataset from kaggle and made an 80/20 test train split. To run the following project, download and place in \Dataset
https://drive.google.com/drive/folders/1tCLHEbImZvoBpLctSq1a84F8n80aBt5t?usp=sharing

The aim of our project is to implement different models for this binary classification task and evaluate the performance. 
For evaluation, we will be using the Normalized Gini Coefficient. Gini is a vital metric in insurance because we care more about segregating high and low risks than predicting losses. A higher gini is going to mean our model is better able to tell between high risk and low risk cases. This will be explained more in chapter 7.
We may also use the auc in places. This has a linear relationship to the Gini: 2*AUC - 1 = Gini.

Required python packages:
numpy
datetime
pandas
sklearn
lightgbm
bayesian_optimization
numba
matplotlib
random
