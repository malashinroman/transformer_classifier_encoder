import os
import time

import irr_tools
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from tqdm import tqdm
import argparse

from transformer_playground.transformer_encoder.clebert import CleBert
from transformer_playground.transformer_encoder.utils import zeroout_experts

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--skip_validation", default=0, type=int)
parser.add_argument("--num_workers", default=0, type=int)
config = parser.parse_args()

# clebert = CleBert(config)
clebert = CleBert(config)
clebert.to(config.device)
