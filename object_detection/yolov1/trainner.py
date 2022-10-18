
from torch.optim import Adam, SGD
import torch.nn as nn
from utils.box import *
from object_detection.yolov1.util.config import Config
import torch

class Trainner:


    def __init__(self, model: nn.Module, opt:Config):
        self.opt = opt
        self.model = model
        # self.optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        self.optimizer = SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.wd)
        self.conf_c = 1
        self.conf_c_noobj = 0.5
        self.cls_c = 1
        self.pos_c = 5



    def train_epoch(self):
        pass


    def train_step(self, img, label):
        out = self.model(img)  # (n c h w) as (n, 30, 7, 7)
        loss = self.calc_loss(out, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def _convert_xywh2xywh(self, box, y, x):
        # 还原xywh的在图像中的真实值
        cx = (box[0] + x) / float(self.opt.grid_x)
        cy = (box[1] + y) / float(self.opt.grid_y)
        w = box[2]
        h = box[3]
        return [cx, cy, w, h]


    def calc_loss(self, predict, labels):
        conf_loss, conf_noobj_loss, pos_loss, cls_loss = 0, 0, 0, 0
        n, c, h, w = predict.shape
        for k in range(n):
            for i in range(self.opt.grid_y):
                for j in range(self.opt.grid_x):
                    latent = predict[k, :, i, j] # 30
                    label = labels[k, :, i, j] # 30
                    p_box1, p_box2, box = latent[:4], latent[5:9], label[:4]

                    p_conf1, p_conf2, conf = latent[4], latent[9], label[4]

                    #obj
                    if conf == 1:
                        box_ = convert_xywh2x1y1x2y2(self._convert_xywh2xywh(box, i, j))
                        p_box1_ = convert_xywh2x1y1x2y2(self._convert_xywh2xywh(p_box1, i, j))
                        p_box2_ = convert_xywh2x1y1x2y2(self._convert_xywh2xywh(p_box2, i, j))
                        iou1 = calc_iou_single(p_box1_, box_)
                        iou2 = calc_iou_single(p_box2_, box_)
                        cls_loss += torch.sum((latent[-self.opt.n_cls:] - label[-self.opt.n_cls:])**2)

                        if iou1 > iou2:
                            pos_loss += torch.sum((p_box1[:2]-box[:2])**2)
                            pos_loss += torch.sum((torch.sqrt(p_box1[2:])-torch.sqrt(box[2:]))**2)
                            conf_loss += (p_conf1 - iou1)**2
                            conf_noobj_loss += (p_conf2-iou2)**2
                        else:
                            pos_loss += torch.sum((p_box2[:2]-box[:2])**2)
                            pos_loss += torch.sum((torch.sqrt(p_box2[2:])-torch.sqrt(box[2:]))**2)
                            conf_loss += (p_conf2 - iou2)**2
                            conf_noobj_loss += (p_conf1-iou1)**2
                    # noobj
                    else:
                        conf_noobj_loss += (p_conf1**2 + p_conf2**2)
        loss = conf_noobj_loss * self.conf_c_noobj + pos_loss * self.pos_c + cls_loss * self.cls_c + conf_loss * self.conf_c
        loss /= float(n)
        # print("loss:", loss)
        return loss





if __name__ == '__main__':
    from object_detection.yolov1.util.config import opt
    from object_detection.yolov1.models.model import MyNet_resnet34
    trainner = Trainner(MyNet_resnet34(), opt)
    p = r"/fastdata/computervision/huangwei/codes/others/a.pth"
    la = r"/fastdata/computervision/huangwei/codes/others/b.pth"
    a = torch.load(p)
    b = torch.load(la)
    loss = trainner.calc_loss(a, b)
    print(loss)
    print(a.shape)

