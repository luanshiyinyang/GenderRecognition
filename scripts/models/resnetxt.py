from torchvision import models
import torch.nn as nn
import torch


def ResNetxt101():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2, bias=True)
    return model


if __name__ == '__main__':
    net = ResNetxt101()
    print(net(torch.randn((32, 3, 224, 224))).size())