import time
import os
from glob import glob

import numpy as np
from PIL import Image

from models.resnet import ResNet50
from models.varg_facenet import varGFaceNet
from models.facenet import facenet
from models.tresnet.tresnet import tresnet


def get_logdir(root_path):
    timestamp = time.strftime('%m%d-%H%M', time.localtime())
    log_dir = os.path.join(root_path, '{}'.format(timestamp))
    return log_dir


def get_mean_std():
    img_h, img_w = 224, 224  # 根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []

    imgs_path = '../dataset/train/'
    imgs_path_list = os.listdir(imgs_path)

    len_ = len(imgs_path_list)
    i = 0
    for item in imgs_path_list:
        img = Image.open(os.path.join(imgs_path, item)).convert('RGB').resize((img_w, img_h))
        img = np.array(img)
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))


def get_exp_num(log_path):
    num = len(os.listdir(log_path))
    return os.path.join(log_path, "exp{}".format(num))


def get_kfold_model(path):
    all_path = []
    for exp in os.listdir(path):
        model_path = glob(os.path.join(path, exp) + '/last*.pth')[0]
        all_path.append(model_path)
    return all_path


def get_model_by_name(name='resnet50'):
    model = None
    if name == 'resnet50':
        model =  ResNet50()
    elif name == 'facenet':
        model = facenet()
    else:
        model = tresnet()
    return model



if __name__ == '__main__':
    get_mean_std()