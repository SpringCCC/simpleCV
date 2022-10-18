# -*- coding: utf-8 -*-
# @Time    : 2022/10/18 10:05
# @Author  : WeiHuang

import torch
import numpy as np


def convert_xywh2x1y1x2y2(box):
    x, y, w, h = box[0], box[1], box[2], box[3]
    x1 = x - w/2.0
    y1 = y - h/2.0
    x2 = x + w/2.0
    y2 = y + h/2.0
    return (x1, y1, x2, y2)



def calc_iou_single(box1, box2):
    """
    :param box1: x1,y1,x2,y2
    :param box2: x1,y1,x2,y2
    :return:
    """
    tl = (max(box1[0], box2[0]), max(box1[1], box2[1]))
    br = (min(box1[2], box2[2]), min(box1[3], box2[3]))
    h = max(br[1]-tl[1], 0)
    w = max(br[0]-tl[0], 0)
    inter = h * w
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # assert area1>0 and area2>0
    iou = inter / (area1+area2-inter)
    return iou

def calc_ious_multi(boxes1, boxes2):
    pass


if __name__ == '__main__':
    b1 = torch.from_numpy(np.asarray([0,0, 5, 5])).cuda()
    b2 = torch.from_numpy(np.asarray([2,2, 10, 10])).cuda()
    iou = calc_iou_single(b1, b2)
    print(iou)