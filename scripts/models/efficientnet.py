import torch
from efficientnet_pytorch import EfficientNet
from torch import nn


def Efficientnet():
    model = EfficientNet.from_pretrained('efficientnet-b5')
    feature = model._fc.in_features
    model._fc = nn.Linear(in_features=feature,out_features=2,bias=True)
    return model


if __name__ == '__main__':
    net = Efficientnet()
    print(net)