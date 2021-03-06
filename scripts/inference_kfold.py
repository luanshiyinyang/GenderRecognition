from argparse import ArgumentParser
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import ttach as tta

from dataset import TestDataset, get_transforms, get_tta_transforms
from utils import get_kfold_model, get_model_by_name, Config

parser = ArgumentParser()
parser.add_argument("--weights", type=str, default="../runs/exp12/")
parser.add_argument("--tta", type=str, default='no')
opt = parser.parse_args()
cfg = Config()

desc_test = os.path.join(cfg.ds_folder, 'test.csv')
_, transform_test = get_transforms(cfg.img_size)
valid_data = TestDataset(desc_test, data_folder=os.path.join(cfg.ds_folder, "test"), transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=cfg.bs, shuffle=False)

models = []
for path in get_kfold_model(opt.weights):
    model = get_model_by_name(cfg.model_name)
    model.load_state_dict(torch.load(path)['state_dict'])
    if opt.tta == 'yes':
        model = tta.ClassificationTTAWrapper(model, get_tta_transforms(), merge_mode='mean')
    models.append(model)

rst = [[] for i in range(len(models))]
for index in range(len(models)):
    net = models[index]
    net.to("cuda")
    net.eval()

    files = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, filename = x.cuda(), y
            out = net(x)
            _, pred = torch.max(out.data, 1)
            rst[index].extend(list(pred.cpu().numpy()))
            files.extend(list(filename.numpy()))
    rst[index] = np.array(rst[index])


# sum pred
final_rst = stats.mode(np.array(rst))[0][0]
submit = pd.DataFrame({'id': files, 'label': final_rst})
submit.to_csv("submit.csv", encoding="utf8", index=False)
