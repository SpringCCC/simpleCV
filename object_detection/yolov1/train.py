import sys
sys.path.append("...")
from utils.vis import Vis
from fire import Fire
from .data.dataset import MyDataset
from .util.config import opt
from torch.utils.data import DataLoader
from .models.model import MyNet_resnet34 as mymodel
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm



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
    # prepare train
    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    criterion_mse = nn.MSELoss()

    # train
    for epoch in enumerate(tqdm(opt.epochs)):
        for img, label in enumerate(train_dataloader):
            n = label.shape[0]
            img, label = img.cuda().float(), label.cuda().float()
            label = label.reshape(n, opt.grid_x, opt.grid_y, opt.out_c)
            label = label.permute(0, 3, 1, 2) # (n c h w)
            out = model(img) # (n c h w) as (n, 30, 7, 7)
            loss = calc_loss(out, label, criterion_mse)
            optimizer.zero_grad()
            loss.backeard()
            optimizer.step()

        vt.vis_images()


    # val&save

    # vis


    pass


def calc_loss(predict, label, criterion_mse):
    """
    :param out: (n c h w) c:(x y w h conf x y w h conf c1 c2 ...)
    :param label: follow predict
    :param criterion_mse:
    :return:
    """



if __name__ == '__main__':
    # Fire()
    train()