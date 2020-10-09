from argparse import ArgumentParser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import ttach as tta
from tqdm import tqdm

from models.resnet import ResNet50
from models.varg_facenet import varGFaceNet
from data_loader import TestDataset


parser = ArgumentParser()
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--tta", type=str, default='no')
opt = parser.parse_args()

desc_test = '../dataset/test.csv'
# normMean = [0.5960974, 0.45659876, 0.39084694]
# normStd = [0.25935432, 0.23155987, 0.22708039]
normMean = [0.5961039, 0.45659694, 0.39085034]
normStd = [0.25910342, 0.23129477, 0.22679278]
transform_test = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normMean, std=normStd)
])

valid_data = TestDataset(desc_test, data_folder="../dataset/test", transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=8, shuffle=False)

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
    net = tta.ClassificationTTAWrapper(net, transforms)

rst = []
files = []
for x, y in tqdm(test_loader):
    x, filename = x.cuda(), y
    out = net(x)
    _, pred = torch.max(out.data, 1)
    rst.extend(list(pred.cpu().numpy()))
    files.extend(list(filename.numpy()))

submit = pd.DataFrame({'id': files, 'label': rst})
submit.to_csv("submit.csv", encoding="utf8", index=False)