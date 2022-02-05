from KNN import *

'''
    用于对第一个任务的完成
'''

# 读入数据并数据处理、存储
train = np.array(pd.read_csv('train_data.csv'))
train_data = train[:,:-1]
val = np.array(pd.read_csv('val_data.csv'))
vali_data = val[:,:-1]
test_data = np.array(pd.read_csv('test_data.csv'))
train_data, vali_data, test_data = ProcessingData(train_data, vali_data, test_data)
train_lable = np.squeeze(train[:,-1:])
pd.DataFrame(np.c_[train_data, train[:,-1:]]).to_csv('processing_train_data.csv')
vali_lable = np.squeeze(val[:,-1:])
pd.DataFrame(np.c_[vali_data, val[:,-1:]]).to_csv('processing_val.csv')




k=1
def task1():
    '''
    用于task1的函数
    '''
    # 载入模型
    model1 = KNN()
    # 传入参数
    model1.fit(train_data, train_lable, Euclidean)
    # 交叉验证得出k
    model1.validation(vali_data,vali_lable, range(1,30), "Euclidean")
    # 输入k的值
    # k = int(input())
    prediction1 = model1.predict(test_data, k)
    pd.DataFrame(np.c_[test_data, prediction1]).to_csv('task1_test_Euclidean.csv')
    model1.fit(train_data, train_lable, Manhattan)
    model1.validation(vali_data,vali_lable, range(1,30), "Manhattan")
    # 输入k的值
    # k = int(input())
    prediction2 = model1.predict(test_data, k)
    pd.DataFrame(np.c_[test_data, prediction2]).to_csv('task1_test_Manhattan.csv')

    model1.fit(train_data, train_lable, Chebyshev)
    model1.validation(vali_data,vali_lable, range(1,30), "Chebyshev")
    # 输入k的值
    # k = int(input())
    prediction3 = model1.predict(test_data, k)
    pd.DataFrame(np.c_[test_data, prediction3]).to_csv('task1_test_Chebyshev.csv')

task1()



