import os
import numpy as np
import mindspore.nn as nn
import mindspore.dataset as ds
import matplotlib.pyplot as plt
from mindspore.train.callback import  ModelCheckpoint, CheckpointConfig
from mindspore import Tensor, Model, load_checkpoint
from mindspore.nn import Accuracy
from LeNet import LeNet
from GooLeNet import GooLeNet
from cross_entropy_loss import *
from dataset_create import *


# 一些路径参数
model_path = "./model"  # 模型路径
image_data_path ="./cifar-10-binary/cifar-10-batches-bin"  # 数据集路径


# 定义网络模型
# net = LeNet(num_classes=10)
net = GooLeNet(num_classes=10)
# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义优化函数
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
# 保存模型
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# ckpoint = ModelCheckpoint(prefix="checkpoint_LeNet", config=config_ck)
ckpoint = ModelCheckpoint(prefix="checkpoint_GooLeNet", config=config_ck) 

# 载入模型
# load_checkpoint("LeNet.ckpt", net=net)
load_checkpoint("GooLeNet.ckpt", net=net)
net_loss = cross_entropy_loss()
model = Model(net, net_loss, metrics={"Accuracy": Accuracy()})
ds_eval = dataset_create(10000, os.path.join(image_data_path, "test"))
acc = model.eval(ds_eval, dataset_sink_mode=False)
print("{}".format(acc))
# 随意从测试数据集选择32张图片并且预测它们
# 蓝色代表分类正确，红色代表分类错误
ds = dataset_create(32, os.path.join(image_data_path, "test"))
ds_test = ds.create_dict_iterator()
data = next(ds_test)
images = data["image"].asnumpy()
labels = data["label"].asnumpy()
output = model.predict(Tensor(data['image']))
prediction = np.argmax(output.asnumpy(), axis=1)
images = np.add(images,1 * 0.1307 / 0.3081)
images = np.multiply(images, 0.3081)
index = 1
for i in range(len(labels)):
    plt.subplot(4, 8, i+1)
    color = 'blue' if prediction[i] == labels[i] else 'red'
    plt.title("pre:{}".format(prediction[i]), color=color)
    img = np.squeeze(images[i]).transpose((1,2,0))
    plt.imshow(img)
    plt.axis("off")
    if color == 'red':
        index = 0
        print("[{}, {}] is incorrect, and it is identified as {}, the correct value should be {}".format(int(i/8)+1, i%8+1, prediction[i], labels[i]))
if index:
    print("Great! There are no errors!")
plt.savefig("Map.jpg")
plt.show()
plt.close()