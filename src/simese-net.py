import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import optim
from torch.autograd import Variable
from torch.utils import data

from siamese_net_19 import SiameseNetwork


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01 * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def default_list_reader(file_list):
    imgList = []
    with open(file_list, 'r') as file:
        for line in file.readlines():
            imgshortList = []
            img_path1, img_path2, label = line.strip().split(' ')
            imgshortList.append(img_path1)
            imgshortList.append(img_path2)
            imgshortList.append(label)
            imgList.append(imgshortList)
    return imgList


def img_loader(path):
    img = Image.open(path)
    return img


def train(train_dataloader, forward_pass, criterion, optimizer, epoch):
    running_loss = 0.0
    iteration_number = 0

    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = Variable(img0), Variable(img1), Variable(label)
        optimizer.zero_grad()
        output1, output2 = forward_pass(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()

        print("Epoch: {}, current iter: {}/{}\n Current loss {}\n".format(epoch, i, len(train_dataloader),
                                                                          loss_contrastive.data[0]))
        running_loss += loss_contrastive.data[0]
    return running_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        # margin = args.batch_size*3
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                              2))
        return loss_contrastive


class train_ImageList(data.Dataset):
    def __init__(self, file_list, transform=None, list_reader=default_list_reader, train_loader=img_loader):
        # self.root      = root
        self.imgList = list_reader(file_list)
        self.transform = transform
        self.train_loader = train_loader

    def __getitem__(self, index):
        [img_path1, img_path2, target] = self.imgList[index]
        img1 = self.train_loader(img_path1)
        img2 = self.train_loader(img_path2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([target], dtype=np.float32))

    def __len__(self):
        return len(self.imgList)


BASE_DIR = os.path.dirname(__file__)
train_txt = os.path.join(BASE_DIR, 'data', 'my_train.txt')

transformer = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), ])
data_set = train_ImageList(file_list=train_txt, transform=transformer)
train_data_loader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=True, num_workers=16)

forward_pass = SiameseNetwork()
print(forward_pass)

criterion = ContrastiveLoss()
optimizer = optim.Adam(forward_pass.get_parameters(), lr=0.01)

for epoch in range(0, 2):
    running_loss = train(train_data_loader, forward_pass, criterion, optimizer, epoch)
