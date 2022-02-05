import math

import numpy as np

from KNN import *

# 计算马氏距离的梯度
def gradient(train_data, train_lable, A, lr = 0.001):
    '''
    用D矩阵存储exp（-d^2）的所有数值
    用d向量存储sum D[k] for k !+ i
    用X矩阵存储所有(train_data[i]-train_data[j])
    用P矩阵存储pij
    用p向量存储sum p[k] for k != i
    '''
    le = len(train_data)
    D = np.zeros([le, le])
    d = []
    X = np.empty([le, le, 1, 4])
    P = np.zeros([le, le])
    p = np.zeros(le)
    for i in range(le):
        # 计算D矩阵 和 X矩阵
        # i == i 时Dii = 0， Xii = 0
        D[i, i] = 0.0
        X[i, i] = 0.0
        for j in range(i + 1, le):
            X[i, j] = (train_data[i] - train_data[j]).reshape([1,4])
            X[j, i] = -X[i, j].reshape([1,4])
            temp = A @ (X[i, j].reshape(-1, 1))
            D[i, j] = math.exp(-np.sum(np.square(temp)))
            D[j, i] = D[i, j]
        # 计算d向量
        di = np.sum(D[i])
        d.append(di)
        # 用于计算P矩阵 和 p向量
        for j in range(le):
            if i == j:
                P[i,j] = 0
            else:
                P[i,j] = D[i,j] / d[i]
            if train_lable[i] == train_lable[j]:
                p[i] += P[i,j]
    sum3 = 0.0
    # 用于计算梯度式子
    for i in range(le):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(le):
            l = X[i,j].reshape([-1,1])
            M = (l @ l.T) * P[i,j]
            sum1 += M
            if train_lable[i] == train_lable[j]:
                sum2 += M
        sum1 *= p[i]
        sum3 = sum3 + (sum1 - sum2)
    gradient = 2 * A @ sum3
    return gradient, np.sum(p)



def Gredient_Descent_batch(train_data, train_lable, A, lr = 0.0001):
    '''
    用于梯度下降学习矩阵A
    :param lr: 学习率，默认0.0001
    :return: 矩阵A， 梯度下降历史
    '''
    print("graient begin")
    epoch = 1
    # 用于记录梯度值的历史，以便画图
    histroy = []
    for j in range(epoch):
        # lr /= math.sqrt(j+1)
        train_data = train_data
        train_lable = train_lable
        gd, sum = gradient(train_data, train_lable, A)
        if sum >= 0.00001:
            A += lr * gd
            histroy.append(sum)
        else:
            break
        print(j+1," batch")
    print("A=", A)
    print("gradient finish")
    return A, histroy


cnt = 1
lrs = [0.4]
#用于选择合适的K，并测试
# 用于选择合适的学习率
def test(A_low_dimemsion = 2):
    '''
    用于选择出较好的学习率
    :param A_low_dimemsion: 用于指定矩阵A的低维
    :return: 不同学习率学习出的矩阵A
    '''
    # 用于存放生成的A
    As= []
    # 用于存放用实验的学习率
    for lr1 in lrs:
        print('learnig rate =',lr1)
        # 初始化矩阵A
        # A = np.random.uniform(size=(A_low_dimemsion, 4))
        A = np.full(shape=(2,4),fill_value=0.01)
        A, histroy = Gredient_Descent_batch(train_data, train_lable, A, lr = lr1)
        As.append(A)
        plt.plot(range(1, len(histroy) + 1), histroy , "r", ms=10)
        plt.xticks(rotation=45)
        plt.xlabel("Numbers of Epoch")
        plt.ylabel("f(A)")
        plt.title("Increament Map for lr = " + str(lr1))
        plt.savefig("Increament_Map_for_lr=" + str(lr1) + ".jpg" )
        plt.close()
    return As


def train_and_test(As):
    accuracys = []
    f1sc = []
    for i in range(len(As)):
        print("learning rate =",lrs[i])
        A = np.array(As[i])
        model = KNN()
        model.fit(train_data, train_lable, Mahalanobis_distance, flag=0, A=A)
        accuracy, f1_sc = model.validation(vali_data, vali_lable, range(1, 30),"lr="+str(lrs[i])+"_Mahalanobis")
        
        k = 20
        print("Please input the right k:")
        a = input(k)
        accuracys.append(accuracy[a])
        f1sc.append(f1_sc[a])
        prediction1 = model.predict(test_data, k)
        pd.DataFrame(np.c_[test_data, prediction1]).to_csv('task2_test_Mahalanobis_lr=' + str(lrs[i]) + '_k=%d.csv'%k)
    return accuracys, f1sc


train = np.array(pd.read_csv('train_data.csv'))
train_data = train[:, :-1]
val = np.array(pd.read_csv('val_data.csv'))
vali_data = val[:, :-1]
test_data = np.array(pd.read_csv('test_data.csv'))
train_data, vali_data, test_data = ProcessingData(train_data, vali_data, test_data)
vali_lable = np.squeeze(val[:, -1:])
train_lable = np.squeeze(train[:, -1:])

#
As = test(2)
for i in range(len(As)):
    pd.DataFrame(As[i]).to_csv("task2_lr=%d"%0.4 + "_A.csv")


accuracy, f1_scores = train_and_test(As)

plt.plot(lrs, accuracy , "r", ms=10)
plt.xticks(rotation=45)
plt.xlabel("Learning rate")
plt.ylabel("Accuracy")
plt.title("Accuracy Map for different learning rate2")
plt.savefig("Accuracy_Map_for_different_learning_rate2.jpg" )
plt.close()


plt.plot(lrs, f1_scores, "r", ms=10)
plt.xticks(rotation=45)
plt.xlabel("Learning rate")
plt.ylabel("f1_scores")
plt.title("f1_scores Map for different learning rate2")
plt.savefig("f1_scores_Map_for_different_learning_rate2.jpg" )
plt.close()








