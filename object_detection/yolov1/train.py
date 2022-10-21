import sys
import os
project = 'simpleCV'
sys.path.append(os.getcwd().split(project)[0]+project)
from utils.vis import Vis
from fire import Fire
from object_detection.yolov1.data.dataset import MyDataset
from object_detection.yolov1.util.config import opt
from torch.utils.data import DataLoader
from object_detection.yolov1.models.model import MyNet_resnet34 as mymodel
from object_detection.yolov1.trainner import Trainner
from torch.optim import Adam
from tqdm import tqdm
import torch
from utils.cv_imgs import *
from utils.vis import *

def train(**kwargs):
    # update params
    opt._update_params(kwargs)
    # build dataset
    train_dataset = MyDataset(opt.dataset_dir, mode='train')
    val_dataset = MyDataset(opt.dataset_dir, mode='val')
    train_dataloader = DataLoader(train_dataset, opt.bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, opt.bs, shuffle=False)
    # build model
    model = mymodel(opt)
    model.cuda()
    vt = Vis(opt)

    # train
    trainner = Trainner(model, opt)
    for epoch in tqdm(range(opt.epochs)):
        model.train()
        for i, (img, label) in enumerate(tqdm(train_dataloader)):
            img, label = img.cuda().float(), label.cuda().float()
            label = label.reshape(-1, opt.grid_x, opt.grid_y, opt.out_c)
            label = label.permute(0, 3, 1, 2)  # (n c h w)
            trainner.train_step(img, label)
            # visual
            if i%opt.vis_freq==0:
                img0 = img[0]
                dets = model.predict(img0[None]) # (x1,y1,x2,y2,score,cls)
                ori_img = reverse_ToTensor(img0)
                render_det_img = vt.draw_objectdetect_rect(ori_img.copy(), dets, opt.GL_CLASSES)
                label0 =label[0]
                label0 = model._reverse_pos(label0) # chw--hwc
                label0 = label0.reshape(-1, opt.out_c)# h*w, c
                dets = label0[:, :5]
                index_0 = np.where(dets[:, 4]==1)[0]
                cls = label0[:, 10:].argmax(axis=1).reshape(-1, 1)
                dets = dets[index_0]
                cls = cls[index_0]
                det0 = np.concatenate([dets, cls],axis=1)
                render_ori_img = vt.draw_objectdetect_rect(ori_img.copy(), det0, opt.GL_CLASSES)
                render_img = np.stack([render_ori_img, render_det_img])
                vt.vis_images(render_img, win=opt.win)


        # vt.vis_images()


    # val&save

    # vis





def check_fuc(trainner: Trainner, model):

    x = torch.rand(5,3,448,448)
    a = model(x)
    labels = torch.zeros(5, 30, 7, 7)
    loss = trainner.calculate_loss(a, labels)
    print(loss)
    print(a.shape)


if __name__ == '__main__':
    # Fire()
    train()

