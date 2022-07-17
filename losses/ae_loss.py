import torch.nn as nn


class AutoencoderLoss(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.device = self.config.device

    def forward(self, data, net_output):
        gt = data["cifar_env_response"].to(self.device)
        output = net_output["restored_resp"]

        loss = (gt - output).abs().mean()
        # loss = self.mse(gt, output)
        return loss
