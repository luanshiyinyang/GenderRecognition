import random
import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import numpy as np
from tqdm import tqdm

from utils import Config, get_model_by_name
from dataset import get_transforms

plt.style.use('fivethirtyeight')
cfg = Config()


def label_analysis():
    df = pd.read_csv(os.path.join(cfg.ds_folder, "train.csv"), encoding="utf8")
    df['label'].value_counts().plot(kind='bar')
    plt.show()


def pic_analysis():
    df = pd.read_csv(os.path.join(cfg.ds_folder, 'train.csv'), encoding="utf8")
    data_folder = os.path.join(cfg.ds_folder, 'train')
    files, labels = df.values[:, 0], df.values[:, 1]
    sample_num = 10
    random_index = random.sample(list(range(len(files))), sample_num)
    files = files[random_index]
    labels = labels[random_index]

    # plot
    plt.figure(figsize=(12, 6))
    for i in range(sample_num):
        plt.subplot(2, 5, i + 1)
        plt.imshow(Image.open(os.path.join(data_folder, str(files[i]) + ".jpg")).convert('RGB'))
        plt.title("label {}".format(labels[i]))

    plt.show()


def plot_top_loss(k=10):
    df = pd.read_csv(os.path.join(cfg.ds_folder, 'new_valid.csv'), encoding='utf8')
    folder = os.path.join(cfg.ds_folder, 'train')
    _, tfms = get_transforms(cfg.img_size)

    model = get_model_by_name(cfg.model_name)
    model.load_state_dict(torch.load(glob("../runs/exp15/best*")[0])['state_dict'])
    model.to("cuda:0")
    model.eval()

    error_list = []
    error_index = []
    for i in tqdm(range(len(df))):
        with torch.no_grad():
            filename = os.path.join(folder, str(df['id'][i]) + ".jpg")
            label = df['label'][i]
            img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            img = tfms(image=img)['image'].astype('float32').transpose(2, 0, 1)
            img = torch.from_numpy(img.reshape(-1, 3, cfg.img_size, cfg.img_size)).to("cuda:0")
            pred = model(img)
            _, pred_label = torch.max(pred, dim=1)
            yes = (pred_label.cpu() == label).item()
            if yes:
                continue
            else:
                error_score = pred.cpu().numpy().squeeze()[pred_label]
                error_list.append(error_score)
                error_index.append(i)
    ind = np.argpartition(np.array(error_list), -k)[-k:]
    error_img_index = np.array(error_index)[ind]

    # plot
    plt.figure(figsize=(12, 6))
    for i in range(k):
        plt.subplot(2, 5, i + 1)
        plt.imshow(Image.open(os.path.join(folder, str(df['id'][error_img_index[i]]) + ".jpg")).convert('RGB'))
        print("filenaame is", str(df['id'][error_img_index[i]]) + ".jpg")
        plt.title("label {}".format(df['label'][error_img_index[i]]))
    plt.show()


if __name__ == '__main__':
    plot_top_loss()
