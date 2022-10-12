from torch import nn
from torch.autograd import Variable
from torch.functional import F
import torch


def MaskedMean(x, mask, dim):
    mask = mask.int().unsqueeze(-1)
    return (x * mask).sum(dim) / mask.sum(dim)


def MaskedSum(x, mask, dim):
    mask = mask.int().unsqueeze(-1)
    return (x * mask).sum(dim)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.8, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.BCE = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, x, y):
        logpt = -self.BCE(x, y)
        pt = torch.exp(logpt)
        loss = - (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        else:
            assert 0, 'No such reduction for FocalLoss'
