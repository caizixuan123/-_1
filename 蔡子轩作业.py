# 导入库
# 数据分析和探索
import pandas as pd
import numpy as np
import random as rnd

import matplotlib.pyplot as plt



np.set_printoptions(threshold = 99)     #  threshold表示输出数组的元素数目


#获取数据
train = np.loadtxt('HTRU_2_train.csv',delimiter = ',')
test = np.loadtxt('HTRU_2_test.csv',delimiter = ',')
print(test.shape)
#print(train)
#print(test)


#集合体现
#combine = [train[:,:-1], test]
classLabels = train[...,-1]
combine = train[:,:-1]
combine = np.append(combine,test,axis = 0)



#训练集观察
data_0 = train[train[:,-1] == 0]
data_1 = train[train[:,-1] == 1]
#print(data_0)
#print(data_1)
#print(combine)
plt.scatter(data_0[...,0],data_0[...,1],linewidth=0.1,color = "red")
plt.scatter(data_1[...,0],data_1[...,1],linewidth=0.1,color = 'blue')
plt.show()



data_0_mean = data_0.mean(axis = 0)
data_1_mean = data_1.mean(axis = 0)
for i in range(2):
    plt.figure(i)
    plt.bar([0,1],[data_0_mean[i],data_1_mean[i]])
    plt.show
    
    
#非线性逻辑回归
#处理数据集
#数据有两个特征我们将数据升维处理
#添加常数列
train_1 = np.insert(train[:,:-1],0,[1],axis = 1)
test_1 = np.insert(test,0,[1],axis = 1)

#添加高次项
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
#print(train_1)


#线性映射
train_min = train_1.min(axis = 0)
test_min = test_1.min(axis = 0)
train_max = train_1.max(axis = 0)
test_max = test_1.max(axis = 0)

for i in range(1,len(train_1[0])):
    train_1[...,i] = (train_1[...,i]-train_min[i])/(train_max[i] - train_min[i])
    test_1[...,i] = (test_1[...,i]-test_min[i])/(test_max[i] - test_min[i])
    
    
#test_1 = test_1[...,1:]
#train_1 = np.insert(train_1,6,classLabels,axis = 1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def grad_descent(dataMathIn,classLabels):
#    把数组类型转化成矩阵类型
    dataMatrix = np.mat(dataMathIn)
#    转化成矩阵类型并进行转置
    labelMat = np.mat(classLabels).transpose()
    
    
#    获取特征矩阵的特征个数和样本个数
    m, n = np.shape(dataMatrix)
#    构建相应的1矩阵
    weights = np.ones((n, 1))
#    print(weights)
    weights_1 = np.zeros((n,1))
    alpha = 0.75
    lamda = 0.75
    maxCycle = 10000
    weights_1[0] = 1
    for i in range(1,len(weights_1)):
        weights_1[i] = 1 * (1 - alpha * lamda / m)
#    print(weights_1 * weights)
#    每一个样本带入公式进行计算推导出相关系数值
    for i in range(maxCycle):
#        if i > 1 and type(weights_1) != 'numpy.matrix':
#            weights_1 = np.mat(weights_1)
        h = sigmoid(dataMatrix * weights)
        weights = weights_1 * np.array(weights) - alpha / m * dataMatrix.transpose() * (h - labelMat)
    return weights


def function(weights):
    def function_1(x):
        y = []
#        for i in x[...,:-1]:
        for i in x:
            if (np.mat(i) * np.mat(weights)) >= 0:
                y.append(1)
            else:
                y.append(0)
#        for i in range(len(x[...,-1])):
#            if x[...,-1][i] == y[i]:
#                temp += 1
#        return temp/len(x)
        return y
    return function_1

weights = grad_descent(train_1,classLabels)
print(weights)
#print(type(weights))
function_1 = function(weights)
list_1 = function_1(test_1)
print(len(list_1))
name = ['id']
test = pd.DataFrame(columns = name,data = list_1)
test.to_csv('D:/test4.csv')
