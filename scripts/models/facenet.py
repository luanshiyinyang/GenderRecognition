import torch
from facenet_pytorch import InceptionResnetV1


def facenet():
    # model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=2)
    model = InceptionResnetV1(pretrained='casia-webface', classify=True, num_classes=2)
    return model


if __name__ == '__main__':
    net = facenet()
    print(net(torch.randn((32, 3, 160, 160))).size())