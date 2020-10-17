import torch
import torch.nn as nn


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1 - lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs * label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs * label, dim=1)
        return loss


class JointLoss(nn.Module):

    def __init__(self, first=nn.CrossEntropyLoss(),
                 second=LabelSmoothSoftmaxCE(),
                 first_weight=0.5,
                 second_weight=0.5):
        super(JointLoss, self).__init__()
        self.loss1 = first
        self.loss2 = second
        self.coef1 = first_weight
        self.coef2 = second_weight

    def forward(self, logits, labels):
        return self.loss1(logits, labels) * self.coef1 + self.loss2(logits, labels) * self.coef2
