import numpy as np

import pandas as pd

import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm

import matplotlib.pyplot as mp

import open_file as op

data_class = op.dataset('./train.csv', './test.csv')
train, train_labels, test = data_class.put_data()

# 训练svm模型---基于线性核函数0.32
# model = svm.SVC(kernel='linear', class_weight='balanced')
# model.fit(train_x, train_y)

# 训练svm模型---基于线性核函数0.35
# model = svm.SVC(kernel='linear')
# model.fit(train_x, train_y)

# 训练svm模型---基于多项式核函数
model = svm.SVC(kernel='poly', degree=3)
# model.fit(train_x, train_y)

# 训练svm模型---基于径向基核函数0.32
# model = svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=1)
# model.fit(train_x, train_y)

# # 训练svm模型---基于径向基核函数0.31
model = svm.SVC(kernel='rbf')
# print(train, train_labels)
model.fit(train, train_labels)

# 预测
pred_test_y = model.predict(test)
op.Document_Storage(pred_test_y)
# bg = sm.classification_report(test_labels, pred_test_y)
# print('分类报告：', bg, sep='\n')