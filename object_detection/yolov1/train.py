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


def train(**kwargs):
    # update params
    opt._update_params(kwargs)
    # build dataset
    train_dataset = MyDataset(opt.dataset_dir, mode='train')
    val_dataset = MyDataset(opt.dataset_dir, mode='val')
    train_dataloader = DataLoader(train_dataset, opt.bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, opt.bs, shuffle=False)
    # build model
    model = mymodel()
    model.cuda()
    vt = Vis(opt)

    # train
    trainner = Trainner(model, opt)
    for epoch in tqdm(range(opt.epochs)):
        model.train()
        for img, label in tqdm(train_dataloader):
            img, label = img.cuda().float(), label.cuda().float()
            # label = label.reshape(-1, opt.grid_x, opt.grid_y, opt.out_c)
            # label = label.permute(0, 3, 1, 2)  # (n c h w)
            # trainner.train_step(img, label)
            img0 = img[0]
            model.predict(img0[None])

        # vt.vis_images()


    # val&save

    # vis




def calc_loss(predict, label):
    """
    :param out: (n c h w) c:(x y w h conf x y w h conf c1 c2 ...)
    :param label: follow predict
    :param criterion_mse:
    :return:
    """

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

