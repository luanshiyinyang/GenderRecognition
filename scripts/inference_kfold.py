from argparse import ArgumentParser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import ttach as tta

from data_loader import TestDataset
from utils import get_kfold_model, get_model_by_name

parser = ArgumentParser()
parser.add_argument("--weights", type=str, default="../runs/exp12/")
parser.add_argument("--tta", type=str, default='no')
opt = parser.parse_args()

desc_test = '../dataset/test.csv'
normMean = [0.59610313, 0.45660403, 0.39085752]
normStd = [0.25930294, 0.23150486, 0.22701606]
transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])
valid_data = TestDataset(desc_test, data_folder="../dataset/test", transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=32, shuffle=False)

models = []
for path in get_kfold_model(opt.weights):
    model = get_model_by_name('facenet')
    model.load_state_dict(torch.load(path)['state_dict'])
    if opt.tta == 'yes':
        transforms = tta.Compose(
            [
                tta.HorizontalFlip()
            ]
        )
        model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
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
final_rst = stats.mode(np.array(rst))[0][0]
submit = pd.DataFrame({'id': files, 'label': final_rst})
submit.to_csv("submit.csv", encoding="utf8", index=False)