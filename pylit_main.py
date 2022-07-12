import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import models
from losses import AutoencoderLoss
from prepare_data_loader import prepare_data_loader
from utils import zeroout_experts

wandb_logger = WandbLogger(project="pytorch_lightning_test")


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self._args = args
        clebert_class = models.__dict__[args.model]
        self.clebert = clebert_class(args)
        self.clebert.to(args.device)
        self.ae_loss_func = AutoencoderLoss(args)

    def forward(self, x):
        output = self.clebert(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        corrupted, indexes = zeroout_experts(train_batch["cifar_env_response"], 0.6)
        output = self.clebert(corrupted, indexes)
        loss = self.ae_loss_func(train_batch, output)
        self.log("train_loss", loss)
        print(batch_idx)
        return loss

    def validation_step(self, val_batch, batch_idx):
        corrupted, indexes = zeroout_experts(val_batch["cifar_env_response"], 0.6)
        output = self.clebert(corrupted, indexes)
        loss = self.ae_loss_func(val_batch, output)
        self.log("val_loss", loss)
        print(batch_idx)


# data

# train_dataloader, eval_dataloader = prepare_data_loader(config)
# ataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])
#
# train_loader = DataLoader(mnist_train, batch_size=32)
# val_loader = DataLoader(mnist_val, batch_size=32)
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--skip_validation", default=0, type=int)
parser.add_argument("--skip_training", default=1, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--load_checkpoint", default="best_net500.pkl", type=str)
parser.add_argument("--model", default="CLEBERT", type=str)
args = parser.parse_args()
#
# model
model = LitAutoEncoder(args)

train_dataloader, eval_dataloader = prepare_data_loader(args)
# training
trainer = pl.Trainer(
    gpus=1, num_nodes=1, precision=16, limit_train_batches=0.5, logger=wandb_logger
)
trainer.fit(model, train_dataloader, eval_dataloader)
