from argparse import ArgumentParser
import warnings
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from runx.logx import logx

from dataset import TrainDataset, get_transforms
from utils import get_exp_num, get_model_by_name, Config, set_seed

set_seed()
warnings.filterwarnings('ignore')
parser = ArgumentParser()
parser.add_argument("--pretrained", type=str, default=None)
opt = parser.parse_args()


# 超参数设置
cfg = Config()
EPOCH = cfg.epochs
BATCH_SIZE = cfg.bs
LR = cfg.lr
IMG_SIZE = cfg.img_size
# 定义是否使用GPU
device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
print(device)
if cfg.exp_name != 'None':
    logdir = os.path.join("../runs/", cfg.exp_name)
else:
    logdir = get_exp_num("../runs/")
logx.initialize(logdir, coolname=True, tensorboard=True)
cfg.save_config(os.path.join(logdir, 'cfg.yaml'))
start_epoch = 0


# 数据加载
desc_train = os.path.join(cfg.ds_folder, 'new_train.csv')
desc_valid = os.path.join(cfg.ds_folder, 'new_valid.csv')


transform_train, transform_test = get_transforms(cfg.img_size)


train_data = TrainDataset(desc_train, data_folder=os.path.join(cfg.ds_folder, 'train'), transform=transform_train)
valid_data = TrainDataset(desc_valid, data_folder=os.path.join(cfg.ds_folder, 'train'), transform=transform_test)

# 构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


net = get_model_by_name(cfg.model_name)
if opt.pretrained:
    net.load_state_dict(torch.load(opt.pretrained)['state_dict'])
net.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = Ranger(net.parameters(), lr=LR, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, last_epoch=start_epoch-1, step_size=10, gamma=0.1)


def train(epoch):
    net.train()
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    for step, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        out = net(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out.data, 1)
        total += y.size(0)
        correct += (pred == y).squeeze().sum().cpu().numpy()
        train_loss += loss.item()

        if step % 100 == 0:
            print("epoch", epoch, "step", step, "loss", loss.item())

    train_acc = correct / total
    print("train accuracy", train_acc)
    logx.metric('train', {'loss': train_loss, 'accuracy': train_acc}, epoch=epoch)


def valid(epoch):
    net.eval()
    valid_loss = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for step, data in enumerate(valid_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = criterion(out, y)

            _, pred = torch.max(out.data, 1)
            valid_loss += loss.item()
            total += y.size(0)
            correct += (pred == y).squeeze().sum().cpu().numpy()
    valid_acc = correct / total
    print("valid accuracy", valid_acc)
    logx.metric('val', {'loss': valid_loss, 'accuracy': valid_acc}, epoch=epoch)
    return valid_acc


for i in range(start_epoch, start_epoch + EPOCH):
    scheduler.step()
    train(i)
    valid_acc = valid(i)
    logx.save_model({'state_dict': net.state_dict(), 'last_epoch': i}, metric=valid_acc, epoch=i, higher_better=True, delete_old=True)


