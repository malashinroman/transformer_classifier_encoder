import torch.nn as nn


class AutoencoderLossL1(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.device = self.config.device

    def forward(self, net_output, gt):
        # __import__('pudb').set_trace()
        output = net_output["restored_resp"]
        loss = (gt - output).abs().mean()
        # loss = self.mse(gt, output)
        return loss
