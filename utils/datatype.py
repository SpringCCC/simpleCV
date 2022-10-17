import torch
import numpy as np


def toNumpy(x):
    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()


def toTensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cuda()
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return None