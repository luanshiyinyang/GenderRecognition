from argparse import ArgumentParser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import ttach as tta
from tqdm import tqdm

from models.resnet import ResNet50
from data_loader import TestDataset

parser = ArgumentParser()
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--tta", type=str, default='yes')
opt = parser.parse_args()


desc_test = '../dataset/new_valid.csv'
normMean = [0.5960974, 0.45659876, 0.39084694]
normStd = [0.25935432, 0.23155987, 0.22708039]
transform_test = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])
valid_data = TestDataset(desc_test, data_folder="../dataset/train", transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=4, shuffle=False)

net = ResNet50()
net.load_state_dict(torch.load(opt.weights)['state_dict'])
net.to("cuda")
net.eval()
if opt.tta == 'yes':
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            # tta.Scale(scales=[1, 2, 4]),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    net = tta.ClassificationTTAWrapper(net, transforms)

rst = []
for x, _ in tqdm(test_loader):
    x = x.cuda()
    out = net(x)
    _, pred = torch.max(out.data, 1)
    rst.extend(list(pred.cpu().numpy()))
label = list(pd.read_csv("../dataset/new_valid.csv", encoding="utf8")['label'])
print(sum(np.array(rst) == np.array(label))/len(rst))

