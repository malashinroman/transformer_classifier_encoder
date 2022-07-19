import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import losses
import models
from prepare_data_loader import prepare_data_loader
from script_manager.func.add_needed_args import smart_parse_args
from utils import zeroout_experts

# __import__("pudb").set_trace()


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self._args = args
        clebert_class = models.__dict__[args.model]
        self.clebert = clebert_class(args)
        self.clebert.to(args.device)
        # self.loss_function = AutoencoderLoss(args)
        loss_function_type = losses.__dict__[args.loss]
        self.loss_function = loss_function_type(args)

    def forward(self, x):
        output = self.clebert(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._args.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        corrupted, indexes = zeroout_experts(
            train_batch["cifar_env_response"], self._args.zeroout_prob
        )
        output = self.clebert(corrupted, indexes)
        loss = self.loss_function(train_batch, output)
        self.log("train_loss", loss)
        print(batch_idx)
        return loss

    def validation_step(self, val_batch, batch_idx):
        corrupted, indexes = zeroout_experts(
            val_batch["cifar_env_response"], self._args.zeroout_prob
        )
        output = self.clebert(corrupted, indexes)
        loss = self.loss_function(val_batch, output)
        self.log("val_loss", loss)
        print(batch_idx)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--load_checkpoint", default="best_net500.pkl", type=str)
parser.add_argument("--losses", default="AE_MSE_LOSS", type=str)
parser.add_argument("--model", default="CLEBERT", type=str)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--skip_training", default=1, type=int)
parser.add_argument("--skip_validation", default=0, type=int)
parser.add_argument("--loss", default="AE_MSE_LOSS", type=str)
parser.add_argument("--zeroout_prob", default=0.15, type=float)
parser.add_argument("--lr", default=1e-3, type=float)

args = smart_parse_args(parser)

if args.wandb_project_name is not None:
    wandb_logger = WandbLogger(project=args.wandb_project_name)
else:
    wandb_logger = True
# model
model = LitAutoEncoder(args)

train_dataloader, eval_dataloader = prepare_data_loader(args)

# training
trainer = pl.Trainer(
    gpus=1,
    num_nodes=1,
    precision=16,
    limit_train_batches=0.5,
    logger=wandb_logger,
    max_epochs=args.epochs,
)
trainer.fit(model, train_dataloader, eval_dataloader)
