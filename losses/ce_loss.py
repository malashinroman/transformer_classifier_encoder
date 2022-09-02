"""program to calculate the cross entropy loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

"""class that implements cross entropy for transformers"""


class CrossEntropyLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, net_output, gt):
        output = net_output["restored_resp"]
        target = gt.argmax(dim=2).to(self.config.device)
        target = target.view(-1)
        output = output.view(-1, output.shape[-1])

        return self.criterion(output, target)
