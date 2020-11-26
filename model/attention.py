# @File : attention.py 
# @Time : 2019/10/21 
# @Email : jingjingjiang2017@gmail.com 

import os

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SelfAttention(nn.Module):
    """
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_dim, activation='relu'):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8,
                           kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8,
                           kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
        """
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1,
                           width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1,
                           width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width,
                                           height)  # B * C * W * H

        out = self.gamma * self_attetion + x

        return out


class AttentionBlock(nn.Module):
    def __init__(self, F_hand, F_feature, F_int):
        super(AttentionBlock, self).__init__()
        self.W_hand = nn.Sequential(
            nn.Conv2d(F_hand, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_feature = nn.Sequential(
            nn.Conv2d(F_feature, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, hand_x, x):
        hand_x_1 = self.W_hand(hand_x)
        x1 = self.W_feature(x)
        psi = self.relu(hand_x_1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionBlockV2(nn.Module):
    def __init__(self, F_hand, F_feature, F_int):
        super(AttentionBlockV2, self).__init__()
        self.W_hand = nn.Sequential(
            nn.Conv2d(F_hand, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_feature = nn.Sequential(
            nn.Conv2d(F_feature, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, hand_x, x):
        hand_x = self.W_hand(hand_x)
        x = self.W_feature(x)
        psi = self.relu(hand_x + x)
        psi = self.psi(psi)

        return x * psi
