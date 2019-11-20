import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def open_csv(CSV_train_path, CSV_test_path):
    '''
    开启csv文件
    :param CSV_train_path:训练集路径
    :param CSV_test_path:测试集路径
    :return:训练集和测试集
    '''
    global train_pd
    global test_pd
    train_pd = pd.read_csv(filepath_or_buffer=CSV_train_path, header=None, keep_default_na=False)
    test_pd =  pd.read_csv(filepath_or_buffer=CSV_test_path, header=None, keep_default_na=False)
    return train_pd, test_pd
def Separation_train(train_pd):
    '''
    将训练集数据与类别分离
    :param train_pd:训练集
    :return:训练集，类别
    '''
    global labels
    global train
    train = train_pd.loc[:,0:12]
    labels = train_pd[13]
    return train,labels
def frequency(data):
    '''
    归一化处理
    :param data:原数据集
    :return:归一化数据集
    '''
    data = data.astype('float')
    data = (data - data.min()) / (data.max() - data.min())
    return data
def findindex(org, x):
    '''
    :param org:
    :param x:
    :return:
    '''
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result
def Missing_value_processing(data):
    '''
    缺失值填补
    :param data:
    :return:
    '''
    for column in list(data.columns):
        results = findindex(data[column],'?')
        if results != []:
            for result in results:
                # data.loc[result, column] = data[data[column] != '?'][column].astype("float").mean()
                # print(int(data[column].mode()))
                data.loc[result, column] = int(data[column].mode())
    return data
class dataset():
    def __init__(self, CSV_train_path, CSV_test_path):
        train_pd, test_pd = open_csv(CSV_train_path, CSV_test_path)

        train_pd = train_pd.drop(columns=[0, 5, 7, 8, 10, 11, 12])
        test_pd = test_pd.drop(columns=[0, 5, 7, 8, 10, 11, 12])

        train_all, self.test = Missing_value_processing(train_pd), Missing_value_processing(test_pd)
        # train_data, self.test = frequency(train_all), frequency(test_all)
        self.train, self.labels = Separation_train(train_all)

        a = self.train.columns
        sum = 1
        for i in a:
            for j in a:
                if j >= i:
                    self.test[13 + sum] = np.array(self.test[j]).astype(int) * np.array(self.test[i]).astype(int)
                    self.train[13 + sum] = np.array(self.train[j]).astype(int) * np.array(self.train[i]).astype(int)
                    sum += 1


        # data, test_pd = open_csv(CSV_train_path, CSV_test_path)
        # data = Missing_value_processing(data)
        #
        # # print(data.corr())
        # data = data.drop(columns=[0, 5, 7, 8, 10, 11, 12])
        #
        # x_train, x_test = train_test_split(data, test_size=0.2, random_state=1)
        #
        # self.train, self.train_labels = Separation_train(x_train)
        # self.test, self.test_labels = Separation_train(x_test)
        #
        # a = train.columns
        # sum = 1
        # for i in a:
        #     for j in a:
        #         if j >= i:
        #             self.test[13 + sum] = np.array(self.test[j]).astype(int) * np.array(self.test[i]).astype(int)
        #             self.train[13 + sum] = np.array(self.train[j]).astype(int) * np.array(self.train[i]).astype(int)
        #             sum += 1

    def put_data(self):

        train = np.array(self.train)
        train_labels = np.array(self.labels).reshape((7194, 1))
        test = np.array(self.test)

        return train, train_labels, test

    # def put_data(self):
    #     train = np.array(self.train).reshape((1, 7194, 13))
    #     labels = np.array(self.labels).reshape((7194, 1))
    #     test = np.array(self.test).reshape((1, 1798, 13))
    #     print(train.shape, labels.shape, test.shape)
    #     return torch.FloatTensor(train), torch.FloatTensor(labels), torch.FloatTensor(test)

def Document_Storage(test_arr):
    test_lables = pd.DataFrame(test_arr)
    test_lables.to_csv('test_lables.csv')
# def Box_figure(dataset, label)
#     dataset[dataset['T'] == label].boxplot()
#     plt.title(f'label={label}')
#     plt.show()
