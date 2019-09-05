import dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torch.utils.data.dataloader as DataLoader
import torchvision.transforms as transforms
class CustomDataset(Dataset):
    def __init__(self, label_file_path):
        with open(label_file_path, 'r') as f:
            # (image_path(str), image_label(str))
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
            
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = transforms.Compose([transforms.ToTensor()])
        label = int(label)
        return img, label
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
train_data=CustomDataset('labels.txt')
for i, item in enumerate(train_data):
        data, label = item
        print('data:', data)
        print('label:', label)
