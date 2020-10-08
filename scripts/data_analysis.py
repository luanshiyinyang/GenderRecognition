import random
import os

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def label_analysis():
    df = pd.read_csv("../dataset/train.csv", encoding="utf8")
    df['label'].value_counts().plot(kind='bar')
    plt.show()


def pic_analysis():
    df = pd.read_csv("../dataset/train.csv", encoding="utf8")
    data_folder = "../dataset/train/"
    files, labels = df.values[:, 0], df.values[:, 1]
    sample_num = 10
    random_index = random.sample(list(range(len(files))), sample_num)
    files = files[random_index]
    labels = labels[random_index]

    # plot
    plt.figure(figsize=(12, 6))
    for i in range(sample_num):
        plt.subplot(2, 5, i+1)
        plt.imshow(Image.open(os.path.join(data_folder, str(files[i])+".jpg")).convert('RGB'))
        plt.title("label {}".format(labels[i]))
    plt.show()


if __name__ == '__main__':
    pic_analysis()
