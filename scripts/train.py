import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from runx.logx import logx

from model import ResNet50
from data_loader import TrainDataset
from optimizer import RAdam
from utils import get_logdir

# 超参数设置
EPOCH = 50
BATCH_SIZE = 32
LR = 0.001

logx.initialize(get_logdir("../runs/"), coolname=True, tensorboard=True)

# 数据加载
desc_train = '../dataset/new_train.csv'
desc_valid = '../dataset/new_valid.csv'

transform_train = transforms.Compose([
    transforms.RandomCrop((200, 200), padding=2),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
train_data = TrainDataset(desc_train, data_folder="../dataset/train/", transform=transform_train)
valid_data = TrainDataset(desc_valid, data_folder="../dataset/train/", transform=transform_test)

# 构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net = ResNet50()
net.to(device)
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.AdamW(net.parameters(), lr=LR)
optimizer = RAdam(net.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_acc = 0


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
    logx.metric('train', {'loss': train_loss}, epoch=epoch)
    train_acc = correct / total
    print("train accuracy", train_acc)


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
    logx.metric('val', {'loss': valid_loss}, epoch=epoch)
    valid_acc = correct / total
    print("valid accuracy", valid_acc)
    return valid_acc


for i in range(EPOCH):
    train(i)
    scheduler.step()
    valid_acc = valid(i)
    logx.save_model({'state_dict': net.state_dict()}, metric=best_acc, epoch=i, higher_better=True, delete_old=True)
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(net.state_dict(), 'weights.pth')

