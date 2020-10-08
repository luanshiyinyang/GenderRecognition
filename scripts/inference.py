import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ResNet50
from data_loader import TestDataset
import pandas as pd


desc_test = '../dataset/test.csv'
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

valid_data = TestDataset(desc_test, data_folder="../dataset/test", transform=transform_test)
test_loader = DataLoader(dataset=valid_data, batch_size=32, shuffle=False)

net = ResNet50()
net.load_state_dict(torch.load("weights.pth"))
net.to("cuda")
net.eval()

rst = []
files = []
for x, y in test_loader:
    x, filename = x.cuda(), y
    out = net(x)
    _, pred = torch.max(out.data, 1)
    rst.extend(list(pred.cpu().numpy()))
    files.extend(list(filename.numpy()))

# submit = pd.read_csv("../dataset/test.csv", encoding="utf8")
submit = pd.DataFrame({'id': files, 'label': rst})
submit.to_csv("submit.csv", encoding="utf8", index=False)