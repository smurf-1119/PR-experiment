import numpy as np
import pandas as pd
import math as ma

# 读取数据，并进行分割
train = np.array(pd.read_csv("train_data.csv", header=None))
train_data = train[:,1:]
train_lable = train[:,0]
test = np.array(pd.read_csv("test_data.csv",header=None))
test_data = test[:,1:]
test_lable = test[:,0]
train_data1 = train_data[train_lable==1]
train_data2 = train_data[train_lable==2]
train_data3 = train_data[train_lable==3]
num = []
num.append(train_data1.shape[0])
num.append(train_data2.shape[0])
num.append(train_data3.shape[0])
num_all = train.shape[0]
# 计算先验概率
p_class = [num[i]/num_all for i in range(len(num))]

# 计算对于某一个类的某一个特征的类条件概率密度
def singleclass_singlefeature_class_probility(train_data, test_data):
    mean = np.mean(train_data)
    std = np.std(train_data)
    t1 = (ma.sqrt(2*ma.pi)*std)
    prediction = []
    for data in test_data:
        t2 = ma.exp(-(data - mean)**2 / (2 * (std ** 2)))
        prediction.append(t2 / t1)
    return np.array(prediction)

# 计算对于某一个类的类条件概率密度
def singleclass_class_probility(train_data, test_data):
    dimension = 13
    prediction = []
    for i in range(dimension):
        prediction.append(singleclass_singlefeature_class_probility(train_data[:,i], test_data[:,i]))
    return np.array(prediction)

# 计算测试数据对于每一个类的每一个特征的类条件概率密度
prediction1 = singleclass_class_probility(train_data1, test_data).T
prediction2 = singleclass_class_probility(train_data2, test_data).T
prediction3 = singleclass_class_probility(train_data3, test_data).T
pd.DataFrame(prediction1).to_csv("class1_ClassConditionalPdfforEachFeature.csv", index = False, header = "class1")
pd.DataFrame(prediction2).to_csv("class2_ClassConditiona2PdfforEachFeature.csv", index = False, header = "class2")
pd.DataFrame(prediction3).to_csv("class3_ClassConditiona3PdfforEachFeature.csv", index = False, header = "class3")

# 计算测试数据对于每一个类的类条件概率密度
prediction1_1 =np.array([np.prod(prediction1[i,:]) for i in range(58)])
prediction2_2 = np.array([np.prod(prediction2[i,:]) for i in range(58)])
prediction3_3 = np.array([np.prod(prediction3[i,:]) for i in range(58)])
pre = np.hstack([prediction1_1.reshape([-1,1]), prediction2_2.reshape([-1,1]), prediction3_3.reshape(-1,1)])
pd.DataFrame(pre).to_csv("ClassConditionalPdf.csv", index = False, header = ['class1', 'class2', 'class3'])

test_prediction_lable = []
test_prediction_pdf = []
for i in range(len(prediction1_1)):
    # 计算每一个测试数据对于每一个类的联合概率密度
    temp = [prediction1_1[i] * p_class[0], prediction2_2[i] * p_class[1], prediction3_3[i] * p_class[2]]
    # 进行归一化,得到后验概率
    temp = temp / np.sum(temp)
    test_prediction_pdf.append(temp)
    # 找出概率最大的标号,其所属类别即为测试数据预测的标签
    lable = np.argmax(temp) + 1
    test_prediction_lable.append(lable)
test_prediction_pdf = np.array(test_prediction_pdf)
test_ = np.array(test_prediction_lable).reshape([-1,1])
for i in range(3):
    test_ = np.hstack([test_, test_prediction_pdf[:,i].reshape([-1,1])])
pd.DataFrame(test_).to_csv("test_result.csv", header= ["predict_lable", "class1", "class2", "class3"], index=False)

correct_num = 0
# 计算准确率
for i in test_prediction_lable == test_lable:
    if i:
        correct_num += 1
accuracy = correct_num / test_lable.shape[0]
print(accuracy)
