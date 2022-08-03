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

    def forward(self, data, net_output):

        target = data["cifar_env_response"].argmax(dim=2).to(self.config.device)
        input = net_output["restored_resp"]
        target = target.view(-1)
        input = input.view(-1, input.shape[-1])

        return self.criterion(input, target)
