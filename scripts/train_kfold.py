from argparse import ArgumentParser
import warnings
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from runx.logx import logx

from dataset import TrainDataset, get_transforms
from losses import JointLoss
from utils import get_exp_num, get_model_by_name, Config

warnings.filterwarnings('ignore')
parser = ArgumentParser()
parser.add_argument("--pretrained", type=str, default=None)
opt = parser.parse_args()
log_dir = get_exp_num("../runs/")
cfg = Config()
# 超参数设置
EPOCH = cfg.epochs
BATCH_SIZE = cfg.bs
LR = cfg.lr
IMG_SIZE = cfg.img_size


transform_train, transform_test = get_transforms(cfg.img_size)


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


for k in range(5):
    print("training for fold {}".format(k))
    start_epoch = 0
    logx.initialize(os.path.join(log_dir, 'fold_{}'.format(k)), coolname=True, tensorboard=True)
    # 数据加载
    desc_train = os.path.join(cfg.ds_folder, 'new_train_{}.csv'.format(k))
    desc_valid = os.path.join(cfg.ds_folder, 'new_valid_{}.csv'.format(k))

    train_data = TrainDataset(desc_train, data_folder=os.path.join(cfg.ds_folder, "train/"), transform=transform_train)
    valid_data = TrainDataset(desc_valid, data_folder=os.path.join(cfg.ds_folder, "train/"), transform=transform_test)

    # 构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = get_model_by_name(cfg.model_name)
    if opt.pretrained:
        net.load_state_dict(torch.load(opt.pretrained)['state_dict'])
    net.to(device)
    # 定义损失函数和优化方式
    criterion = JointLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, last_epoch=start_epoch - 1, T_max=EPOCH, eta_min=1e-10)

    for i in range(start_epoch, start_epoch + EPOCH):
        train(i)
        scheduler.step()
        valid_acc = valid(i)
        logx.save_model({'state_dict': net.state_dict(), 'epoch': i}, metric=valid_acc, epoch=i, higher_better=True, delete_old=True)


