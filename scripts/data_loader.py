import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class TrainDataset(Dataset):

    def __init__(self, desc_file, data_folder, transform=None):
        self.all_data = pd.read_csv(desc_file).values
        self.data_folder = data_folder
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.all_data[index, 0], self.all_data[index, 1]
        img = Image.open(os.path.join(self.data_folder, str(img)+".jpg")).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.all_data)


class TestDataset(Dataset):

    def __init__(self, desc_file, data_folder, transform=None):
        self.all_data = pd.read_csv(desc_file).values
        self.data_folder = data_folder
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_data[index, 0]
        img = Image.open(os.path.join(self.data_folder, str(filename)+".jpg")).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.all_data)



