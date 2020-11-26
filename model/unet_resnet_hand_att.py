# @File : unet_resnet_hand_att.py 
# @Time : 2019/10/21 
# @Email : jingjingjiang2017@gmail.com

import os

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

from model.unet_resnet import UNetResNet18
from model.attention import AttentionBlock
from opt import *


class UNetResnetHandAtt(nn.Module):
    def __init__(self, n_classes=2):
        super(UNetResnetHandAtt, self).__init__()
        self.base_model = UNetResNet18()
        if args.dataset == 'EPIC':
            self.base_model.load_state_dict(torch.load(os.path.join(
                args.exp_path, f'epic/unet_resnet/ckpts/model_epoch_174.pth'),
                map_location='cpu'))
        else:
            self.base_model.load_state_dict(torch.load(os.path.join(
                args.exp_path, f'adl/unet_resnet/ckpts/model_epoch_1077.pth'),
                map_location='cpu'))
        
        print(f'finish loading {args.dataset} model.')
        
        self.base_model.cuda(device=args.device_ids[0])
        
        self.conv_1x1 = nn.Conv2d(1, 2, kernel_size=1, stride=1)
        
        self.att_block = AttentionBlock(F_hand=2, F_feature=64, F_int=32)
        
        self.out = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64, n_classes, kernel_size=1,
                                           stride=1))
    
    def forward(self, x, hand_x):
        _, f1 = self.base_model(x, with_output_feature_map=True)
        hand_x = self.conv_1x1(hand_x)
        
        x = self.att_block(hand_x, f1)
        x = self.out(torch.cat([f1, x], dim=1))
        
        return x


if __name__ == '__main__':
    model = UNetResnetHandAtt()
    model.cuda()
    
    # summary(model, input_size=(3, 224, 320))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数总数: {total_params}')  # 参数总数: 19341058
    
    input = Variable(torch.randn(8, 3, 224, 320)).cuda()
    # input = Variable(torch.randn(8, 512, 7, 10)).cuda()
    feature_h = Variable(torch.randn(8, 1, 224, 320)).cuda()
    
    out = model(input, feature_h)
    # out = model(input)
