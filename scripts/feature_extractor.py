import torch
import torch.nn as nn
from utils import get_model_by_name


def get_model(pretrained=None):
    model = get_model_by_name('facenet')
    model.load_state_dict(torch.load(pretrained)['state_dict'])
    new_model = nn.Sequential(*list(model.children())[:-1])
    return new_model


if __name__ == '__main__':
    net = get_model("../../runs/exp11/best_checkpoint_ep13.pth")
    print(net(torch.randn((32, 3, 200, 200))).size())
