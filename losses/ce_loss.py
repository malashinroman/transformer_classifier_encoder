'''program to calculate the cross entropy loss'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

'''class that implements cross entropy for transformers'''

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, input, target):
        return self.criterion(input, target)k
