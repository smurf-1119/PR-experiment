import os
import mindspore.nn as nn
import mindspore.dataset as ds
import matplotlib.pyplot as plt
import mindspore.dataset.vision.c_transforms as vi_transforms
import mindspore.dataset.transforms.c_transforms as c_transforms
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import context, Model
from mindspore.common.initializer import Normal
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
from GooLeNet import GooLeNet
from mindspore.nn import Accuracy
from LeNet import LeNet
from cross_entropy_loss import *
from dataset_create import *



loss = {"step": [], "loss_value": []}
accuracy_eval = {"step": [], "acc": []}

# 训练和测试函数
def train_net( model, epoch_size, data_path, repeat_size, ckpoint_cb):
    '''
    1.载入数据集
    2.保存网络模型和参数
    3.收集损失与准确率信息
    '''
    ds_train = dataset_create(50000, os.path.join(data_path, "train"), 32, repeat_size)
    ds_eval = dataset_create(10000, os.path.join(data_path, "test"))
    config_ck = CheckpointConfig(save_checkpoint_steps=375, keep_checkpoint_max=16)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_"+str('GooLeNet'), directory=model_path, config=config_ck)
    step_loss_acc_info = loss_histroy(model ,ds_eval, loss, accuracy_eval)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(per_print_times=1), step_loss_acc_info], dataset_sink_mode=False)

def test_net(model, data_path):
    ds_eval = dataset_create(10000, os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))

# 训练函数
def train():
    # 训练模型
    train_epoch = 1
    dataset_size = 1
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(model, train_epoch, image_data_path, dataset_size, ckpoint)
    test_net(model, image_data_path)
    # 画每一步的损失值图
    steps = loss["step"]
    loss_value = loss["loss_value"]
    steps = list(map(int, steps))
    loss_value = list(map(float, loss_value))
    plt.plot(steps, loss_value, color="red")
    plt.xlabel("steps")
    plt.ylabel("loss_value")
    plt.title("Change chart of model loss value")
    plt.savefig("Loss_step_map.jpg")
    plt.show()
    plt.close()

    steps = accuracy_eval["step"]
    Accuracy_value = accuracy_eval["acc"]
    steps = list(map(int, steps))
    Accuracy_value = list(map(float, Accuracy_value))
    plt.plot(steps, Accuracy_value, color="red")
    plt.xlabel("steps")
    plt.ylabel("Accuracy")
    plt.title("Change chart of model Accuracy value")
    plt.savefig("Accuracy_step_map.jpg")
    plt.show()
    plt.close()

# 一些路径参数
model_path = "./model"  # 模型路径
image_data_path ="./cifar-10-binary/cifar-10-batches-bin"  # 数据集路径
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# 定义网络模型\
net = LeNet(num_classes=10)
# net = GooLeNet(num_classes=10)
# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义优化函数
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
# 保存模型
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
ckpoint = ModelCheckpoint(prefix="checkpoint_LeNet", config=config_ck)
# ckpoint = ModelCheckpoint(prefix="checkpoint_GooLeNet", config=config_ck)

# 训练
train()   