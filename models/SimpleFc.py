import torch
import torch.nn as nn


class SimpleFc(nn.Module):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        super().__init__()
        self.FcCoder = nn.Sequential(
            nn.Linear(100 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 100 * 10),
        )

    def forward(self, corrupted_responses, indexes=None):
        # flatten responses
        corrupted_responses = torch.flatten(corrupted_responses, 1)
        responses = self.FcCoder(corrupted_responses)
        responses = torch.reshape(responses, (-1, 10, 100))
        return {"restored_resp": responses}
