import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(
            (1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs),
            targets)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# # 针对二分类任务的 Focal Loss
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = torch.tensor(alpha).cuda()
#         self.gamma = gamma
#         self.size_average = size_average
#
#     def forward(self, pred, target):
#         # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
#         pred = nn.Sigmoid()(pred)
#
#         # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
#         pred = pred.view(-1, 1)
#         target = target.view(-1, 1)
#
#         # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
#         pred = torch.cat((1 - pred, pred), dim=1)
#
#         # 根据 target 生成 mask，即根据 ground truth 选择所需概率
#         # 用大白话讲就是：
#         # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
#         # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
#         class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
#         # 这里的 scatter_ 操作不常用，其函数原型为:
#         # scatter_(dim,index,src)->Tensor
#         # Writes all values from the tensor src into self at the indices
#         # specified in the index tensor.
#         # For each value in src, its output index is specified by its index
#         # in src for dimension != dim and by the corresponding value in index
#         # for dimension = dim.
#         class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
#
#         # 利用 mask 将所需概率值挑选出来
#         probs = (pred * class_mask).sum(dim=1).view(-1, 1)
#         probs = probs.clamp(min=0.0001, max=1.0)
#
#         # 计算概率的 log 值
#         log_p = probs.log()
#
#         # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
#         alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
#         alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
#         alpha[:, 1] = alpha[:, 1] * self.alpha
#
#         alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
#
#         # 根据 Focal Loss 的公式计算 Loss
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#
#         # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#
#         return loss
#
#
# # 针对 Multi-Label 任务的 Focal Loss
# class FocalLossMultiLabel(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, size_average=True):
#         super(FocalLossMultiLabel, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.size_average = size_average
#
#     def forward(self, pred, target):
#         criterion = FocalLoss(self.alpha, self.gamma, self.size_average)
#         loss = torch.zeros(1, target.shape[1]).cuda()
#
#         # 对每个 Label 计算一次 Focal Loss
#         for label in range(target.shape[1]):
#             batch_loss = criterion(pred[:, label], target[:, label])
#             loss[0, label] = batch_loss.mean()
#
#         # Loss Function的常规操作
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#
#         return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        """

        :param gamma:
        :param alpha:
        :param size_average:
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))
            # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    import time
    import random
    start_time = time.time()
    maxe = 0
    for i in range(1000):
        x = torch.rand(12800, 2) * random.randint(1, 10)
        x = Variable(x.cuda())
        l = torch.rand(12800).ge(0.1).long()
        l = Variable(l.cuda())

        output0 = FocalLoss(gamma=0)(x, l)
        output1 = nn.CrossEntropyLoss()(x, l)
        a = output0.data.item()
        b = output1.data.item()
        if abs(a - b) > maxe:
            maxe = abs(a - b)
    print('time:', time.time() - start_time, 'max_error:', maxe)

    start_time = time.time()
    maxe = 0
    for i in range(100):
        x = torch.rand(128, 1000, 8, 4) * random.randint(1, 10)
        x = Variable(x.cuda())
        l = torch.rand(128, 8, 4) * 1000  # 1000 is classes_num
        l = l.long()
        l = Variable(l.cuda())

        output0 = FocalLoss(gamma=0)(x, l)
        output1 = nn.NLLLoss2d()(F.log_softmax(x), l)
        a = output0.data.item()
        b = output1.data.item()
        if abs(a - b) > maxe:
            maxe = abs(a - b)
    print('time:', time.time() - start_time, 'max_error:', maxe)

