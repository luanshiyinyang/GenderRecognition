import time
import os
from glob import glob
import random

import numpy as np
from PIL import Image
import yaml
import torch
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from models.resnet import ResNet50
from models.varg_facenet import varGFaceNet
from models.facenet import facenet
from models.inception_resnetv2 import inceptionresnet2
from models.wrn import wrn50


def get_logdir(root_path):
    timestamp = time.strftime('%m%d-%H%M', time.localtime())
    log_dir = os.path.join(root_path, '{}'.format(timestamp))
    return log_dir


def get_mean_std():
    img_h, img_w = 224, 224  # 根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []

    imgs_path = '/home/zhouchen/Datasets/GR/train/'
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
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    num = len(os.listdir(log_path))
    return os.path.join(log_path, "exp{}".format(num))


def get_kfold_model(path):
    all_path = []
    for exp in os.listdir(path):
        model_path = glob(os.path.join(path, exp) + '/best*.pth')[0]
        all_path.append(model_path)
    return all_path


def get_model_by_name(name='resnet50'):
    model_dict = {
        'resnet50': ResNet50(),
        'facenet': facenet(),
        'vargfacenet': varGFaceNet(),
        'inception_resnet': inceptionresnet2(),
        'wrn': wrn50()
    }
    print("use model {}".format(name))
    if name in model_dict.keys():
        model = model_dict[name]
    else:
        raise ValueError("no model like this")
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
        self.config = read_config()
        # exp info
        exp_info = self.config['exp']
        self.exp_name = exp_info['exp_name']
        # training info
        training_info = self.config['training']
        self.lr = training_info['lr']
        self.bs = training_info['bs']
        self.img_size = training_info['img_size']
        self.epochs = training_info['epochs']
        self.model_name = training_info['model_name']
        self.device = "cuda:{}".format(training_info['device'])

        # dataset info
        dataset_info = self.config['dataset']
        self.ds_folder = dataset_info['root']

    def save_config(self, filename):
        assert filename.endswith(".yaml"), 'not a yaml file'
        with open(filename, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)


def set_seed(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def read_tfevents(filepath="../runs/baseline/"):
    filename = glob(filepath + '/events*')[0]

    ea = event_accumulator.EventAccumulator(filename)
    ea.Reload()
    train_loss = [x.value for x in ea.scalars.Items('train/loss')]
    train_accuracy = [x.value for x in ea.scalars.Items('train/accuracy')]
    val_loss = [x.value for x in ea.scalars.Items('val/loss')]
    val_accuracy = [x.value for x in ea.scalars.Items('val/accuracy')]
    return train_loss, train_accuracy, val_loss, val_accuracy


import os
import numpy as np
from scipy.io import loadmat
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def gen_csv_for_imagnet(root_folder='/media/zhouchen/ZC/Dataset/GR2/', subfolder='Training'):
    data_folder = os.path.join(root_folder, subfolder)
    map_dict = {
        'male': 0,
        'female': 1
    }
    ids, labels = [], []
    for category in os.listdir(data_folder):
        files = os.listdir(os.path.join(data_folder, category))
        for f in tqdm(files):
            ids.append(os.path.splitext(f)[0])
            labels.append(map_dict[category.strip()])
    pd.DataFrame({'id': ids, 'label': labels}).to_csv(os.path.join(root_folder, subfolder+'.csv'), encoding='utf8', index=False)


if __name__ == '__main__':
    gen_csv_for_imagnet()



