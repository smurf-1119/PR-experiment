
import matplotlib
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# 计算欧几里得距离
def Euclidean(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

# 计算曼哈顿距离
def Manhattan(a,b):
    return np.sum(np.abs(a-b))

# 计算切比雪夫距离
def  Chebyshev(a,b):
    return np.max(np.abs(a-b))

# 计算马氏距离
def Mahalanobis_distance(xi, A, xj):
    temp = np.squeeze(A @ ((xi - xj).reshape(-1,1)))
    return np.sqrt(np.sum(np.square(temp)))


# 对数据进行预处理
def ProcessingData(train_data, vali_data, test_data):
    # # 数据标准化
    # mean = data.mean(axis=0)
    # std = data.std(axis=0)
    # data = (data - mean)/std
    # 数据归一化
    data = np.vstack((train_data, vali_data, test_data))
    t_max, t_min = data.max(axis=0), data.min(axis=0)
    train_data = (train_data - t_min) / (t_max - t_min)
    vali_data = (vali_data - t_min) / (t_max - t_min)
    test_data = (test_data - t_min) / (t_max - t_min)
    return train_data, vali_data, test_data


def f1_score(TP, FP, TN, FN):
    '''
    :param TP: 真的真
    :param FP: 假的真
    :param TN: 真的假
    :param FN:  假的假
    :return: f1_score
    '''
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    if (TP + FP) != 0:
        precision = (TP) / (TP + FP)
    if (TP + FN) != 0:
        recall = (TP) / (TP + FN)
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1



class KNN():
    def fit(self, train_data, train_lable, metric, K = 9, flag = 1, A = np.array([[1,1,1,1],[1,1,1,1]])):
        '''
        :param train_data: 用于存储训练的数据集
        :param train_lable: 用于存储训练数据的标签
        :param metric:用于指明度量方式
        :param K: 用于指明近邻的个数， 默认为9
        :param flag: 用于指明是否用到矩阵A，1 不用， 0 用，默认不用
        :param A: A 矩阵 用于对 Mahalanobis_distance 方式的度量
        '''
        self.train_data = train_data
        self.train_lable = train_lable
        self.metrics = metric
        self.K = K
        self.flag = flag
        self.A = A
    def k_nearest_neighbor(self, test_data):
        '''
        :param test_data: 要分类的数据
        :return: x的标签
        '''
        # 用于存放test_data和train——data的所有距离
        distances = []
        # 遍历训练数据集里的所有点，计算出所有距离
        for i in range(len(self.train_data)):
            # 不使用矩阵A的度量方式
            if self.flag:
                distance = self.metrics(test_data, self.train_data[i])
            # 使用矩阵A的度量方式
            else:
                distance = self.metrics(test_data, self.A, self.train_data[i])
            distances.append(distance)
        distances = np.array(distances)
        # 得到distances升序的数字序号
        sort_number = distances.argsort()
        # 取对应的前k个lable，分别计算1和0的个数，并选出最多的标签，作为test_data的标签
        count = self.train_lable[sort_number[:self.K]]
        cnt1 = np.sum(count)
        cnt0 = len(count) - cnt1
        if cnt1 > cnt0:
            return 1
        else:
            return 0



    def validation(self, vali_data, vali_lable, test_datas, plot_name):
        '''
        :param vali_data: 交叉验证的数据集
        :param vali_lable: 交叉验证的数据标签
        :param test_datas: 测试的数据集
        :param plot_name: 用于指明存储的图片的名称
        '''
        # le： 数据长度
        le = len(vali_lable)
        # accuracys：用于存放准确率
        accuracys = {}
        # f1_scores：用于存放f1_score
        f1_scores = {}

        # 遍历所有测试数据
        for i in test_datas:
            TP = 0.0
            FP = 0.0
            TN = 0.0
            FN = 0.0
            # 只尝试奇数的k
            self.K = 2 * i - 1
            for j in range(le):
                test_lable = self.k_nearest_neighbor(vali_data[j])
                if test_lable == vali_lable[j]:
                    if test_lable == 1:
                        TP += 1.0
                    else:
                        TN += 1.0
                else:
                    if test_lable == 1:
                        FP += 1.0
                    else:
                        FN += 1.0

            accuracys[str(2*i-1)] = (TP+TN)/le#
            f1_scores[str(2*i-1)] = f1_score(TP, FP, TN, FN)

        print("accuracy=",accuracys)
        print("f1_scores=",f1_scores)
        # K 为坐标画关于准确率的折线图
        x = accuracys.keys()
        y = accuracys.values()
        plt.plot(x, y, "r", ms = 10)
        plt.xticks(rotation=45)
        plt.xlabel("Numbers of K")
        plt.ylabel("accuracy")
        plt.title("Accuracy Map")
        if (self.flag):
            plt.savefig("task1_" + plot_name + "Accuracy_Map.jpg")
        else:
            plt.savefig("task2_" + plot_name + "Accuracy_Map.jpg")
        plt.close()

        # K 为坐标画关于f1_scores的折线图
        x = f1_scores.keys()
        y = f1_scores.values()
        plt.plot(x, y, "r", ms=10)
        plt.xticks(rotation=45)
        plt.xlabel("Numbers of K")
        plt.ylabel("f1_scores")
        plt.title("f1_scores Map")
        if (self.flag):
            plt.savefig("task1_"+ plot_name +"f1_scores Map.jpg")
        else:
            plt.savefig("task2_" + plot_name +"f1_scores Map.jpg")
        plt.close()
        return accuracys, f1_scores


    def predict(self, test_data, k):
        '''
        :param test_data: 测试数据集
        :param k: 选择合适的k
        :return: 预测的标签集
        '''
        self.K = k
        predictions = []
        for i in test_data:
            prediction = self.k_nearest_neighbor(i)
            predictions.append(prediction)
        print(predictions)
        return predictions









