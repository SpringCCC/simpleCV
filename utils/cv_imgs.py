import cv2
import numpy as np
import os
import torch
from utils.datatype import *
"""
windows下安装opencv-python时出现问题
解决办法：下载与python版本对应的opencv.whl文件
pip install xxx.whl
进行安装
"""


def read_img(abs_path):
    return cv2.imdecode(np.fromfile(abs_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def save_img(sv_path, img, post="jpg"):
    """
    :param sv_path:
    :param img:
    :param mode: rgb or gray
    :return:
    """
    if post=="rgb":
        cv2.imencode(".jpg", img)[1].tofile(sv_path)
    elif post=="gray":
        cv2.imencode(".png", img)[1].tofile(sv_path)
    else:
        raise ValueError("check post value, only 'jpg' or 'png'")

def reverse_ToTensor(img):
    assert isinstance(img, torch.Tensor)
    img = img.permute(1, 2, 0)
    img = toNumpy(img)
    img *= 255
    img = img.clip(0, 255)
    return img



