from argparse import ArgumentParser
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import ttach as tta
from tqdm import tqdm

from dataset import TestDataset
from utils import get_model_by_name, Config

parser = ArgumentParser()
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--tta", type=str, default='no')
opt = parser.parse_args()
cfg = Config()
IMG_SIZE = cfg.img_size


desc_test = '../dataset/new_valid.csv'
normMean = [0.59610313, 0.45660403, 0.39085752]
normStd = [0.25930294, 0.23150486, 0.22701606]
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])
valid_data = TestDataset(desc_test, data_folder=os.path.join(cfg.ds_folder, "train"), transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=False)

net = get_model_by_name(cfg.model_name)
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
label = list(pd.read_csv(os.path.join(cfg.ds_folder, "new_valid.csv"), encoding="utf8")['label'])
print(sum(np.array(rst) == np.array(label))/len(rst))

