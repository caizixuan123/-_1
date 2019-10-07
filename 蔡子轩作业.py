# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:51:32 2019

@author: 17704
"""

import pandas as pd
import numpy as np



np.set_printoptions(threshold = 99)     #  threshold表示输出数组的元素数目


#非线性逻辑回归
def Generating_higher(train,test):
    train_1 = np.insert(train[:,:-1],0,[1],axis = 1)
    test_1 = np.insert(test,0,[1],axis = 1)
    temp_1 = train_1[...,1] * train_1[...,1]
    train_1 = np.insert(train_1,3,temp_1,axis = 1)
    temp_4 = test_1[...,1] * test_1[...,1]
    test_1 = np.insert(test_1,3,temp_4,axis = 1)
    temp_2 = train_1[...,1] * train_1[...,2]
    train_1 = np.insert(train_1,4,temp_2,axis = 1)
    temp_5 = test_1[...,1] * test_1[...,2]
    test_1 = np.insert(test_1,4,temp_5,axis = 1)
    temp_3 = train_1[...,2] * train_1[...,2]
    train_1 = np.insert(train_1,5,temp_3,axis = 1)
    temp_6 = test_1[...,2] * test_1[...,2]
    test_1 = np.insert(test_1,5,temp_6,axis = 1)
    
    return train_1,test_1

#线性映射
def Linear_mapping(train_1,test_1):
    tr_min = train_1.min(axis = 0)
    te_min = test_1.min(axis = 0)
    tr_max = train_1.max(axis = 0)
    te_max = test_1.max(axis = 0)

    for i in range(1,len(train_1[0])):
        train_1[...,i] = (train_1[...,i]-tr_min[i])/(tr_max[i] - tr_min[i])
        test_1[...,i] = (test_1[...,i]-te_min[i])/(te_max[i] - te_min[i])
    return train_1,test_1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def grad_descent(dataMathIn,classLabels):
    dataMatrix = np.mat(dataMathIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    weights_1 = np.zeros((n,1))
    alpha = 0.50
    lamda = 0.50
    maxCycle = 10000
    weights_1[0] = 1
    for i in range(1,len(weights_1)):
        weights_1[i] = 1 * (1 - alpha * lamda / m)
#    每一个样本带入公式进行计算推导出相关系数值
    for i in range(maxCycle):
        h = sigmoid(dataMatrix * weights)
        weights = weights_1 * np.array(weights) - alpha / m * dataMatrix.transpose() * (h - labelMat)
    return weights


def function_1(x,weights):
    y = []
    for i in x:
        if (np.mat(i) * np.mat(weights)) >= 0:
            y.append(1)
        else:
            y.append(0)
    return y

train = np.loadtxt('HTRU_2_train.csv',delimiter = ',')#获取训练级
test = np.loadtxt('HTRU_2_test.csv',delimiter = ',')#获取测试集
classLabels = train[...,-1]
train_1,test_1 = Generating_higher(train,test)
train_1,test_1 = Linear_mapping(train_1,test_1)
weights = grad_descent(train_1,classLabels)
list_1 = function_1(test_1,weights)
name = ['id']
test = pd.DataFrame(columns = name,data = list_1)
test.to_csv('D:/test2.csv')