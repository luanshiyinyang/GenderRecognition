import time
import os
from glob import glob

import numpy as np
from PIL import Image
import yaml

from models.resnet import ResNet50
from models.varg_facenet import varGFaceNet
from models.facenet import facenet
from models.resnest.resnest import resnest50


def get_logdir(root_path):
    timestamp = time.strftime('%m%d-%H%M', time.localtime())
    log_dir = os.path.join(root_path, '{}'.format(timestamp))
    return log_dir


def get_mean_std():
    img_h, img_w = 160, 160  # 根据自己数据集适当调整，影响不大
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
        model_path = glob(os.path.join(path, exp) + '/best*.pth')[0]
        all_path.append(model_path)
    return all_path


def get_model_by_name(name='resnet50'):
    model = None
    if name == 'resnet50':
        model =  ResNet50()
    elif name == 'facenet':
        model = facenet()
    elif name == 'resnest':
        model = resnest50()
    else:
        model = resnest50()
    return model


def read_config(config_file="../config/cfg.yaml"):
    current_path = os.path.dirname(__file__)
    config_file = os.path.join(current_path, config_file)
    assert os.path.isfile(config_file), "not a config file"
    print("load config file from", config_file)
    with open(config_file, 'r', encoding="utf8") as f:
        cfg = yaml.safe_load(f.read())
    return cfg


class Config(object):
    def __init__(self):
        config = read_config()

        # training info
        training_info = config['training']
        self.lr = training_info['lr']
        self.bs = training_info['bs']
        self.img_size = training_info['img_size']
        self.epochs = training_info['epochs']
        self.model_name = training_info['model_name']


if __name__ == '__main__':
    import torch
    net = get_model_by_name('resnest')
    print(net(torch.randn((32, 3, 200, 200))).size())