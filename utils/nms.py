# -*- coding: utf-8 -*-
# @Time    : 2022/10/20 16:50
# @Author  : WeiHuang

import numpy as np
from utils.box import bboxes_iou
from torchvision.ops import nms
import torch

def nms_multi_cls(dets, n_cls, thresh=0.1):
    """
    :param dets: (N,6) x1, y2, x2, y2, conf, cls
    :return:
    """
    keeps = []
    boxes = dets[:, :4]
    confs = dets[:, 4]
    cls = dets[:, 5]
    for i in range(n_cls):
        index_i = np.where(cls==i)[0]
        box_i = boxes[index_i]
        score_i = confs[index_i]
        keep = nms_single_cls(box_i, score_i, thresh)
        keeps.extend(index_i[keep])
    return keeps

def nms_single_cls(box, score, thresh):
    order = score.argsort()[::-1]
    keeps = []
    while len(order) > 1:
        keeps.append(order[0])
        box_0 = box[order[0]].reshape(-1, 4)
        box_post = box[order[1:]].reshape(-1, 4)
        iou = bboxes_iou(box_0, box_post).flatten()
        keep = np.where(iou < thresh)[0]
        keep += 1
        order = order[keep]
    if len(order) > 0:
        keeps.append(order[0])
    return keeps