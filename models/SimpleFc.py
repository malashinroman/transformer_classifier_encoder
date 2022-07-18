import torch
import torch.nn as nn


class SimpleFc(nn.Module):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        super().__init__()
        self.FcCoder = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 768),
        )

    def forward(self, corrupted_responses, indexes=None):
        responses = self.FcCoder(corrupted_responses)
        return {"restored_resp": responses}
