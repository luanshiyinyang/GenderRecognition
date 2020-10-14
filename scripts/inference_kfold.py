from argparse import ArgumentParser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats

from models.varg_facenet import varGFaceNet
from data_loader import TestDataset
from utils import get_kfold_model

parser = ArgumentParser()
parser.add_argument("--weight_path", type=str, default="../runs/exp10/")
opt = parser.parse_args()

desc_test = '../dataset/test.csv'
normMean = [0.59610415, 0.4566031, 0.39085707]
normStd = [0.25930327, 0.23150527, 0.22701454]
transform_test = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])
valid_data = TestDataset(desc_test, data_folder="../dataset/test", transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=False)

models = []
for path in get_kfold_model(opt.weight_path):
    model = varGFaceNet()
    model.load_state_dict(torch.load(path)['state_dict'])
    models.append(model)

rst = [[] for i in range(len(models))]
for index in range(len(models)):
    net = models[index]
    net.to("cuda")
    net.eval()

    files = []
    for x, y in tqdm(test_loader):
        x, filename = x.cuda(), y
        out = net(x)
        _, pred = torch.max(out.data, 1)
        rst[index].extend(list(pred.cpu().numpy()))
        files.extend(list(filename.numpy()))
    rst[index] = np.array(rst[index])


# sum pred
final_rst = stats.mode(np.array(rst))[0]

submit = pd.DataFrame({'id': files, 'label': final_rst.reshape(-1)})
submit.to_csv("submit.csv", encoding="utf8", index=False)
