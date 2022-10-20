from utils.nms import nms_multi_cls
import torch
import torch.nn as nn
from torchvision.models import resnet34
from utils.block.CBR import CBR
from utils.box import *
from utils.datatype import *
from object_detection.yolov1.util.config import Config
from torchvision.ops import nms



class MyNet_resnet34(nn.Module):

    def __init__(self, opt: Config):
        super(MyNet_resnet34, self).__init__()
        self.opt = opt
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


# 在自己复现一个经典目标检测项目，练手用
    #   总是记不住stack和concat的用法
    # stack增加一个维度
    # concate 不增加维度
    def predict(self, x):
        """
        :param x: 默认是已经处理好了，可以直接使用模型预测的图像
        :return:
        """
        self.eval()
        with torch.no_grad():
            predict = self.forward(x)
            predict = predict[0]
        c, h, w = predict.shape
        predict = predict.permute(1, 2, 0) # hwc
        predict = toNumpy(predict)
        predict =  self._reverse_pos(predict)
        predict = predict.reshape(-1, c)
        boxes = predict[:, :10].reshape(-1, 5) # (98, 5)x,y,w,h,conf
        boxes[:, :4] = boxes[:, :4].clip(0,1)
        cls = predict[:, 10:] # (49, 20)
        cls = cls.argmax(axis=1) #(49, )
        cls = cls.repeat(2) # #(98, )
        boxes = np.concatenate([boxes, cls], axis=-1) # x, y, w, h, score, cls
        keep = nms_multi_cls(boxes)
        # keep_boxes = boxes[keep]

    def _reverse_pos(self, predict):
        """
        :param predict: hwc c:30
        :return:
        """
        predict = toNumpy(predict)
        h, w, c = predict.shape
        for i in range(h):
            for j in range(w):
                predict[i, j, :4] = convert_xywh2x1y1x2y2(self._convert_xywh2xywh(predict[i, j, :4], i, j, h, w))
                predict[i, j, 5:9] = convert_xywh2x1y1x2y2(self._convert_xywh2xywh(predict[i, j, 5:9], i, j, h, w))
        return predict


    def _convert_xywh2xywh(self, box, y, x, grid_y, grid_x):
        # 还原xywh的在图像中的真实值
        cx = (box[0] + x) / float(grid_x)
        cy = (box[1] + y) / float(grid_y)
        w = box[2]
        h = box[3]
        return [cx, cy, w, h]



#
if __name__ == '__main__':
    # p = r"C:\Users\HuangWei\Desktop\新建文件夹\a.pth"
    p = r"/fastdata/computervision/huangwei/codes/others/a.pth"
    predict = torch.load(p)
    model = MyNet_resnet34(Config())

    predict = predict[0]
    c, h, w = predict.shape
    predict = predict.permute(1, 2, 0) # hwc
    predict = toNumpy(predict)
    predict =  model._reverse_pos(predict)
    predict = predict.reshape(-1, c)
    boxes = predict[:, :10].reshape(-1, 5) # (98, 5)x,y,w,h,conf
    boxes[:, :4] = boxes[:, :4].clip(0,1)
    cls = predict[:, 10:] # (49, 20)
    cls = cls.argmax(axis=1) #(49, )
    cls = cls.repeat(2) # #(98, )
    boxes = np.concatenate([boxes, cls.reshape(-1, 1)], axis=-1) # x, y, w, h, score, cls
    keep = nms_multi_cls(boxes, 20)
    a  = 1



