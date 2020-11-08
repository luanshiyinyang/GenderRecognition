from torchvision.models import wide_resnet50_2
import torch.nn as nn


def wrn50():
    model = wide_resnet50_2(pretrained=True)
    model.fc = nn.Linear(2048, 2)
    return model


if __name__ == '__main__':
    wrn50()