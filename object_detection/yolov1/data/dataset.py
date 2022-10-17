import sys
sys.path.append("..")
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as tsf
from utils.cv_imgs import *
import torch

class MyDataset(Dataset):

    def __init__(self, dataset_dir, seed=None, mode='train', train_val_ratio=0.9, trans=None):
        """
        :param dataset_dir: 数据所在文件夹
        :param seed: 打乱数据所用的随机种子
        :param mode: 数据模型: train val test
        :param train_val_ratio: 训练时，训练集:验证集比例
        :param trans: 图像变换函数
        """
        if seed is None:
            seed = random.randint(0, 65526)
        random.seed(seed)
        self.dataset_dir = dataset_dir
        self.mode = mode
        if mode=='val':
            mode='train'
        img_list_txt = os.path.join(dataset_dir, mode + ".txt")  # 储存图片位置的列表
        label_csv = os.path.join(dataset_dir, mode + ".csv")  # 储存标签的数组文件
        self.img_list = []
        self.label = np.loadtxt(label_csv)  # 读取标签数组文件
        # 读取图片位置文件
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())
        # 在mode=train或val时， 将数据进行切分
        # 注意在mode="val"时，传入的随机种子seed要和mode="train"相同
        self.num_all_data = len(self.img_list)
        all_ids = list(range(self.num_all_data))
        num_train = int(train_val_ratio * self.num_all_data)
        if self.mode == "train":
            self.use_ids = all_ids[:num_train]
        elif self.mode == "val":
            self.use_ids = all_ids[num_train:]
        else:
            self.use_ids = all_ids
        # 储存数据增广函数
        self.trans = trans

    def __getitem__(self, item):
        """
        TODO:
        1. 按顺序依次取出第item个训练数据img及其对应的样本标签label
        2. 图像数据要进行预处理，并最终转换为(c, h, w)的维度，同时转换为torch.tensor
        3. 样本标签要按需要转换为指定格式的torch.tensor
        """
        id = self.use_ids[item]
        label = torch.tensor(self.label[id, :])
        img_path = self.img_list[id]
        img = read_img(img_path) # bgr
        img = img[:, :, ::-1] # rgb
        if self.trans is None:
            trans = tsf.Compose([
                # transforms.Resize((112,112)),
                tsf.ToTensor(),
            ])
        else:
            trans = self.trans
        img = trans(img.copy())  # 图像预处理&数据增广
        return img, label

    def __len__(self):
        """获取数据集数量"""
        return len(self.use_ids)

if __name__ == '__main__':
    # 调试用，依次取出数据看看是否正确
    dataset_dir = r"E:\ImageData\VOCdevkit\VOC2007\voc2012_forYolov1"
    dataset = MyDataset(dataset_dir)
    dataloader = DataLoader(dataset, 2)
    for i, data in enumerate(dataloader):
        # input("press enter to continue")
        print(data[0].shape)

