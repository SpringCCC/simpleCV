from visdom import Visdom
import cv2
from utils.datatype import *

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


    def _check_dets_boxv(self, img, boxes):
        if len(boxes)==0:
            return boxes
        if (boxes<=1).all():
            h, w, c = img.shape
            boxes[:, ::2] *= w
            boxes[:, 1::2] *= h
        boxes = boxes.astype(np.int)
        return boxes


    def draw_objectdetect_rect(self, img, dets, cls_names=None):
        if len(dets)==0:
            return img
        if len(dets[0])==6:
            boxes, scores, clss = dets[:, :4], dets[:, 4], dets[:, 5]
        elif len(dets[0])==5:
            boxes, clss = dets[:, :4], dets[:, 4]
        elif len(dets[0])==4:
            boxes = dets[:, :4]
        boxes = self._check_dets_boxv(img, boxes)
        for box, score, cls in zip(boxes, scores, clss):
            cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (255, 0, 0))
            if cls_names:
                cv2.putText(img, cls_names[int(cls)], tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return img











