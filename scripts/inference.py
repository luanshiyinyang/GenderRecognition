from argparse import ArgumentParser
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import ttach as tta
from tqdm import tqdm

from dataset import TestDataset, get_transforms
from utils import get_model_by_name, Config


cfg = Config()
IMG_SIZE = cfg.img_size


parser = ArgumentParser()
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--tta", type=str, default='no')
opt = parser.parse_args()

desc_test = os.path.join(cfg.ds_folder, 'test.csv')

_, test_tfms = get_transforms(IMG_SIZE)

valid_data = TestDataset(desc_test, data_folder=os.path.join(cfg.ds_folder, "test"), transform=test_tfms)
test_loader = DataLoader(dataset=valid_data, batch_size=cfg.bs, shuffle=False)

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
files = []
for x, y in tqdm(test_loader):
    with torch.no_grad():
        x, filename = x.cuda(), y
        out = net(x)
        _, pred = torch.max(out.data, 1)
        rst.extend(list(pred.cpu().numpy()))
        files.extend(list(filename.numpy()))

submit = pd.DataFrame({'id': files, 'label': rst})
submit.to_csv("submit.csv", encoding="utf8", index=False)