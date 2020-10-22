import os
from torch.utils.data import Dataset
import cv2
import pandas as pd
import albumentations
import numpy as np


class TrainDataset(Dataset):

    def __init__(self, desc_file, data_folder, transform=None):
        self.all_data = pd.read_csv(desc_file).values
        self.data_folder = data_folder
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.all_data[index, 0], self.all_data[index, 1]
        img = cv2.imread((os.path.join(self.data_folder, str(img)+".jpg")))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            res = self.transform(image=img)
            image = res['image'].astype(np.float32)
        else:
            image = img.astype(np.float32)

        image = image.transpose(2, 0, 1)
        return image, label

    def __len__(self):
        return len(self.all_data)


class TestDataset(Dataset):

    def __init__(self, desc_file, data_folder, transform=None):
        self.all_data = pd.read_csv(desc_file).values
        self.data_folder = data_folder
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_data[index, 0]
        img = cv2.imread((os.path.join(self.data_folder, str(filename) + ".jpg")))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            res = self.transform(image=img)
            image = res['image'].astype(np.float32)
        else:
            image = img.astype(np.float32)

        image = image.transpose(2, 0, 1)
        return image, filename

    def __len__(self):
        return len(self.all_data)


def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.CoarseDropout(max_height=int(image_size * 0.15), max_width=int(image_size * 0.15), max_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val



