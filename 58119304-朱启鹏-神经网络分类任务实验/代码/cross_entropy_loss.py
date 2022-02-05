import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import  Tensor
from mindspore import dtype as mstype
from mindspore.train.callback import Callback



class cross_entropy_loss(nn.Cell):
    '''
    定义损失函数类
    '''
    def __init__(self):
        super(cross_entropy_loss, self).__init__()
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean()
        self.one_hot = ops.OneHot()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, ops.shape(logits)[1], self.one, self.zero)
        loss_func = self.cross_entropy(logits, label)[0]
        loss_func = self.mean(loss_func, (-1,))
        return loss_func

class loss_histroy(Callback):
    '''
        定义记录损失历史的类
    '''
    def __init__(self, model, dataset, loss, accuracy_eval):
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.accuracy_eval = accuracy_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        cur_step_num = (cur_epoch_num-1)*1875 + cb_params.cur_step_num
        self.loss["loss_value"].append(str(cb_params.net_outputs))
        self.loss["step"].append(str(cur_step_num))
        if cur_step_num % 125 == 0:
            acc = self.model.eval(self.dataset, dataset_sink_mode=False)
            self.accuracy_eval["step"].append(cur_step_num)
            self.accuracy_eval["acc"].append(acc["Accuracy"])