import os.path
from random import choice

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class Train_dataset(Dataset):
    def __init__(self, interact_train, item_num, trainSet_u):
        super(Train_dataset, self).__init__()
        self.interact_train = interact_train
        self.item_list = range(item_num)
        self.trainSet_u = trainSet_u

    def __len__(self):
        return len(self.interact_train)

    def __getitem__(self, idx):
        entry = self.interact_train.iloc[idx] # 选择第 idx行的数据

        # user, item, negitem
        user = entry.userid
        pos_item = entry.itemid
        neg_item = choice(self.item_list)
        while neg_item in self.trainSet_u[user]: # 如果随机挑选出来的 项目被该用户交互过 则重新挑选
            neg_item = choice(self.item_list)

        return user, pos_item, neg_item


class Test_dataset(Dataset):
    def __init__(self, testSet_u, item_num):
        super(Test_dataset, self).__init__()
        self.testSet_u = testSet_u
        self.user_list = list(testSet_u.keys())
        self.item_num = item_num

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item_list = torch.tensor(list(self.testSet_u[user].keys())) # 第idx个用户交互过的item列表
        tensor = torch.zeros(self.item_num).scatter(0, item_list, 1)  # [0,0,0,1,1,0] 第idx个用户对第四第五的item有交互
        return user, tensor  # 返回用户id 和 表示用户交互过的项目下标的tensor


class Data(object):
    def __init__(self, interact_train, interact_test, user_num, item_num):
        self.interact_train = interact_train
        self.interact_test = interact_test
        self.user_num = user_num
        self.item_num = item_num

        self.user_list = list(range(self.user_num))
        self.item_list = list(range(self.item_num))

        self.userMeans = {}  # mean values of users's ratings
        self.itemMeans = {}  # mean values of items's ratings
        self.globalMean = 0 # 所有user的平均评分

        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)

        self.__generateSet()  # 为两个数据集中的用户和项目分别定义交互矩阵 train/testSet_u[userName][itemName] = rating train/testSet_i[itemName][userName] = rating
        self.__computeItemMean()  # 计算每一项目的平均评分 dict
        self.__computeUserMean()  # 计算每一用户的平均评分 dict
        self.__globalAverage()  # 计算所有用户的平均评分 int

        self.train_dataset = Train_dataset(self.interact_train, self.item_num, self.trainSet_u)
        self.test_dataset = Test_dataset(self.testSet_u, self.item_num)

        user_historical_mask = np.ones((user_num, item_num))
        for uuu in self.trainSet_u.keys():
            item_list = list(self.trainSet_u[uuu].keys()) # uuu交互过的item
            if len(item_list) != 0:
                user_historical_mask[uuu, item_list] = 0

        self.user_historical_mask = torch.from_numpy(user_historical_mask) # user_historical_mask[uuu, item] == 1 表示用户uuu对于item未交互

    def __generateSet(self):
        for row in self.interact_train.itertuples(index=False): #遍历每一行
            userName = row.userid
            itemName = row.itemid
            rating = row.score
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating

        for row in self.interact_test.itertuples(index=False):
            userName = row.userid
            itemName = row.itemid
            rating = row.score
            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating

    def __computeItemMean(self):
        for c in self.item_list:
            self.itemMeans[c] = sum(self.trainSet_i[c].values()) / (len(self.trainSet_i[c]) + 0.00000001)

    def __computeUserMean(self):
        for c in self.user_list:
            self.userMeans[c] = sum(self.trainSet_u[c].values()) / (len(self.trainSet_u[c]) + 0.00000001)

    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total == 0:
            self.globalMean = 0
        else:
            self.globalMean = total / len(self.userMeans)


def data_load(dataset_name, test_dataset, bottom):
    '''
    dataset_name: 数据集的名称或路径，用于指定要加载的数据集。
    test_dataset: 一个布尔值，指示是否加载测试数据集。如果为 True，则加载测试数据集；否则，不加载测试数据集。
    bottom: 一个整数值，用于过滤评分低于指定数值的项目
    '''
    save_dir = 'dataset/' + dataset_name
    if not os.path.exists(save_dir):
        print("dataset is not exist!!!")
        return None

    if test_dataset == True:
        interact_train = pd.read_pickle(save_dir + '/interact_train.pkl')
        interact_test = pd.read_pickle(save_dir + '/interact_test.pkl')
        item_encoder_map = pd.read_csv(save_dir + '/item_encoder_map.csv')
        item_num = len(item_encoder_map)
        user_encoder_map = pd.read_csv(save_dir + '/user_encoder_map.csv')
        user_num = len(user_encoder_map)

        if bottom != None:
            interact_train = interact_train[interact_train['score'] > bottom]
            interact_test = interact_test[interact_test['score'] > bottom]
        return interact_train, interact_test, user_num, item_num
