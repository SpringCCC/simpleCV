import torch
import torch.nn as nn
from torchvision.models import resnet34
from utils.block.CBR import CBR
from utils.box import *


class MyNet_resnet34(nn.Module):

    def __init__(self):
        super(MyNet_resnet34, self).__init__()
        backbone = resnet34(pretrained=True)
        self.in_channel = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-2]) #去掉resnet最后两层(分类头)，原始是用于分类
        self.mix_layer = nn.Sequential(CBR(self.in_channel, self.in_channel * 2, 3, 1, 1),
                                       CBR(self.in_channel * 2, self.in_channel * 2, 3, 2, 1),
                                       CBR(self.in_channel * 2, self.in_channel * 2, 3, 1, 1),
                                       CBR(self.in_channel * 2, self.in_channel * 2, 3, 1, 1))
        # todo 这里可以尝试把全连接层使用conv2d代替生成最终的输出
        self.out_layer = nn.Sequential(nn.Linear(7*7*1024, 4096),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(4096, 7*7*30),
                                       nn.Sigmoid())

    def forward(self, x):
        n = x.shape[0]
        f = self.backbone(x) # n, 512, 14, 14
        f = self.mix_layer(f)# n, 1024, 7, 7
        f = f.reshape(n, -1)
        out = self.out_layer(f)
        out = out.reshape(n, 30, 7, 7)
        return out


    def predict(self):
        pass
