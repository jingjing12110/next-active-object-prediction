# @File : BasicNetwork.py 
# @Time : 2019/10/6 
# @Email : jingjingjiang2017@gmail.com 


from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResNet18(nn.Module):
    def __init__(self, pre_trained=True, require_grad=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pre_trained)

        self.body = [layers for layers in self.model.children()]
        self.body.pop(-1)

        self.body = nn.Sequential(*self.body)

        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 512)
        return x


# from scratch
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        f_gate = self.gate(combined)
        i_gate = self.gate(combined)
        o_gate = self.gate(combined)
        f_gate = self.sigmoid(f_gate)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        cell_helper = self.gate(combined)
        cell_helper = self.tanh(cell_helper)
        cell = torch.add(torch.mul(cell, f_gate), torch.mul(cell_helper, i_gate))
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        output = self.softmax(output)
        return output, hidden, cell

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def initCell(self):
        return Variable(torch.zeros(1, self.cell_size))


class ComputeRNN(nn.Module):
    def __init__(self, in_feature, hidden_size, n_class):
        super(ComputeRNN, self).__init__()
        self.in_feature = in_feature
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.in2hidden = nn.Linear(in_feature + self.hidden_size, self.hidden_size)
        self.hidden2out = nn.Linear(self.hidden_size, self.n_class)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    # #此处input的尺寸为[seq_len,batch,in_feature]
    def forward(self, input, pre_state):
        T = input.shape[0]
        batch = input.shape[1]
        a = Variable(torch.zeros(T, batch, self.hidden_size))  # a-> [T,hidden_size]
        o = Variable(torch.zeros(T, batch, self.n_class))  # o ->[T,n_class]
        predict_y = Variable(torch.zeros(T, batch, self.n_class))
        # pre_state = Variable(torch.zeros(batch, self.hidden_size))  # pre_state=[batch,hidden_size]

        if pre_state is None:
            pre_state = Variable(torch.zeros(batch, self.hidden_size))  # hidden ->[batch,hidden_size]

        for t in range(T):
            # input:[T,batch,in_feature]
            tmp = torch.cat((input[t], pre_state), 1)
            # [batch,in_feature]+[batch,hidden_size]-> [batch,hidden_size+in_featue]
            a[t] = self.in2hidden(tmp)
            # [batch,hidden_size+in_feature]*[hidden_size+in_feature,hidden_size] ->[batch,hidden_size]
            hidden = self.tanh(a[t])

            # 这里不赋值的话就没有代表隐层向前传递
            pre_state = hidden

            o[t] = self.hidden2out(hidden)  # [batch,hidden_size]*[hidden_size,n_class]->[batch,n_class]
            # 由于此次是一个单分类问题，因此不用softmax函数
            if self.n_class == 1:
                predict_y[t] = F.sigmoid(o[t])
            else:
                predict_y[t] = self.softmax(o[t])

        return predict_y, hidden


