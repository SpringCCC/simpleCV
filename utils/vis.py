from visdom import Visdom
import cv2
from .datatype import *
class Vis():

    def __init__(self, opt):
        self.vis = Visdom(env=opt.env, use_incoming_socket=False)

    def vis_images(self, imgs, win):
        assert len(imgs.shape) == 4
        self.vis.images(imgs, win=win, opts=dict(title=win))


    def vis_lines(self, epoch, x, win):
        self.vis.line([x], [epoch], win=win, opts=dict(title=win), update='append')


    def revert_img_for_vis(self, x, mean=0.5, std=0.5):
        assert len(x.shape) == 4
        return x * std + mean




