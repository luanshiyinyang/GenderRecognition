from argparse import ArgumentParser
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from runx.logx import logx

from models.varg_facenet import varGFaceNet
from data_loader import TrainDataset
from optimizer import Ranger
from loss import LabelSmoothSoftmaxCE
from utils import get_exp_num

warnings.filterwarnings('ignore')
parser = ArgumentParser()
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--model", type=str, default="resnet50")
opt = parser.parse_args()


# 超参数设置
EPOCH = 30
BATCH_SIZE = 8
LR = 0.001
IMG_SIZE = 112

logx.initialize(get_exp_num("../runs/"), coolname=True, tensorboard=True)

# 数据加载
desc_train = '../dataset/new_train.csv'
desc_valid = '../dataset/new_valid.csv'
# size for 200
# normMean = [0.5960974, 0.45659876, 0.39084694]
# normStd = [0.25935432, 0.23155987, 0.22708039]
# size for 112
normMean = [0.5961039, 0.45659694, 0.39085034]
normStd = [0.25910342, 0.23129477, 0.22679278]

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])
train_data = TrainDataset(desc_train, data_folder="../dataset/train/", transform=transform_train)
valid_data = TrainDataset(desc_valid, data_folder="../dataset/train/", transform=transform_test)

# 构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net = varGFaceNet()
if opt.pretrained:
    net.load_state_dict(torch.load(opt.pretrained)['state_dict'])
net.to(device)
# 定义损失函数和优化方式
criterion = LabelSmoothSoftmaxCE()
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.AdamW(net.parameters(), lr=LR)
optimizer = Ranger(net.parameters(), lr=LR)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


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

    for step, data in enumerate(valid_loader):
        net.eval()
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


for i in range(EPOCH):
    train(i)
    # scheduler.step()
    valid_acc = valid(i)
    logx.save_model({'state_dict': net.state_dict()}, metric=valid_acc, epoch=i, higher_better=True, delete_old=True)


