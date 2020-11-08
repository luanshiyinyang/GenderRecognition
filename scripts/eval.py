from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import ttach as tta
from tqdm import tqdm

from dataset import TestDataset, get_transforms, get_tta_transforms
from utils import get_model_by_name, Config

parser = ArgumentParser()
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--tta", type=str, default='no')
opt = parser.parse_args()
cfg = Config()
IMG_SIZE = cfg.img_size
device = cfg.device


desc_test = os.path.join(cfg.ds_folder, 'new_valid.csv')

_, test_tfms = get_transforms(IMG_SIZE)
valid_data = TestDataset(desc_test, data_folder=os.path.join(cfg.ds_folder, "train"), transform=test_tfms)
test_loader = DataLoader(dataset=valid_data, batch_size=cfg.bs, shuffle=False)

net = get_model_by_name(cfg.model_name)
net.load_state_dict(torch.load(opt.weights)['state_dict'])
net.to(device)
net.eval()
if opt.tta == 'yes':
    net = tta.ClassificationTTAWrapper(net, get_tta_transforms(), merge_mode='mean')

rst = []
for x, _ in tqdm(test_loader):
    with torch.no_grad():
        x = x.to(device)
        out = net(x)
        _, pred = torch.max(out.data, 1)
        rst.extend(list(pred.cpu().numpy()))
label = list(pd.read_csv(os.path.join(cfg.ds_folder, "new_valid.csv"), encoding="utf8")['label'])
print(sum(np.array(rst) == np.array(label))/len(rst))

