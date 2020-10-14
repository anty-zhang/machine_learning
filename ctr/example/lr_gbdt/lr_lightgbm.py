import lightgbm as lgb

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

print('Load data...')
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]

# print(df_test.head(10))

y_train = df_train['target']  # training label
y_test = df_test['target']  # testing label
X_train = df_train[NUMERIC_COLS]  # training dataset
X_test = df_test[NUMERIC_COLS]  # testing dataset

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# pred_leaf=True: predict and get data on leaves, training data
y_pred = gbm.predict(X_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[:1])

"""
# 训练数据量和树的棵树
(10000, 100)
# 每条训练数据不是gbdt的预测值，而是落在每棵树的哪个叶子节点上
[[59 45 17 17 12 12 18 11 18 18 56 12 12 12 44 42 22 24 24 24 14 39 58 15
  25  3 47  3  3  3  7  5 31 38 38 63  5 61 61 56 15  7 17 51 19 56  0 51
   0 32  2  2 36 46 34 57 48 24 52 42  2 43 62 56 55 51 24 52 16 27 53 14
  37 60 63  1 35 35 49 35 32  3 50  3  4 63 51 15 39 55 48 63 36 25 25 62
  14 26 23 12]]
"""

# 需要将每棵树的特征进行one-hot处理
# transformed_training_matrix的维度为: (N(训练样本数量), num_tress(树的棵树) * num_leafs(每棵树的叶子数))
print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
# 计算每个训练样本在100棵树种的叶子索引，并transformed_training_matrix矩阵对应位置置为1
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1

# 对于测试集也要进行同样的处理
y_pred = gbm.predict(X_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1

# 使用LR进行训练
lr = LogisticRegression(penalty='l2', C=0.05)   # logistic model construction
lr.fit(transformed_training_matrix, y_train)    # fitting the data
y_pred_test = lr.predict_proba(transformed_testing_matrix)   # Give the probability on each label

print(y_pred_test)
NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:, 1]) + (1-y_test)/2 * np.log(1 - y_pred_test[:, 1])))
print("Normalized Cross Entropy " + str(NE))
