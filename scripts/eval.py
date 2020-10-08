import torch
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ResNet50
from data_loader import TestDataset
import pandas as pd
import numpy as np

parser = ArgumentParser()
parser.add_argument("--weights", type=str, default=None)
opt = parser.parse_args()


desc_test = '../dataset/new_valid.csv'
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
valid_data = TestDataset(desc_test, data_folder="../dataset/train", transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=False)

net = ResNet50()
net.load_state_dict(torch.load(opt.weights)['state_dict'])
net.to("cuda")
net.eval()

rst = []
for x, _ in test_loader:
    x = x.cuda()
    out = net(x)
    _, pred = torch.max(out.data, 1)
    rst.extend(list(pred.cpu().numpy()))
label = list(pd.read_csv("../dataset/new_valid.csv", encoding="utf8")['label'])
print(sum(np.array(rst) == np.array(label))/len(rst))

