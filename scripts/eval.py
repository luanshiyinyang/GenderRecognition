from argparse import ArgumentParser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import ttach as tta
from tqdm import tqdm

from models.resnet import ResNet50
from models.varg_facenet import varGFaceNet
from data_loader import TestDataset

parser = ArgumentParser()
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--tta", type=str, default='no')
opt = parser.parse_args()


desc_test = '../dataset/new_valid.csv'
normMean = [0.59610415, 0.4566031, 0.39085707]
normStd = [0.25930327, 0.23150527, 0.22701454]
transform_test = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])
valid_data = TestDataset(desc_test, data_folder="../dataset/train", transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=4, shuffle=False)

net = varGFaceNet()
net.load_state_dict(torch.load(opt.weights)['state_dict'])
net.to("cuda")
net.eval()
if opt.tta == 'yes':
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
        ]
    )
    net = tta.ClassificationTTAWrapper(net, transforms, merge_mode='mean')

rst = []
for x, _ in tqdm(test_loader):
    x = x.cuda()
    out = net(x)
    _, pred = torch.max(out.data, 1)
    rst.extend(list(pred.cpu().numpy()))
label = list(pd.read_csv("../dataset/new_valid.csv", encoding="utf8")['label'])
print(sum(np.array(rst) == np.array(label))/len(rst))

