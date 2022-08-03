import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.optim import Adam

import losses
import models
from prepare_data_loader import prepare_data_loader
from script_manager.func.add_needed_args import smart_parse_args
from script_manager.func.wandb_logger import write_wandb_dict
from utils import zeroout_experts

sys.path.append(".")
from local_config import WEAK_CLASSIFIERS


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(",")]


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--fixed_zero_exp_num", default=0, type=int)
parser.add_argument("--load_checkpoint", default=None, type=str)
parser.add_argument("--loss", default="AE_MSE_LOSS", type=str)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--model", default="Clebert", type=str)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--optimizer", default="Adam", type=str)
parser.add_argument("--skip_training", default=0, type=int)
parser.add_argument("--skip_validation", default=0, type=int)
parser.add_argument("--use_static_files", default=1, type=int)
parser.add_argument("--zeroout_prob", default=0.15, type=float)
parser.add_argument("--dataset", default="cifar100", type=str)
parser.add_argument(
    "--weak_classifier_folder",
    type=str,
    default=os.path.join(
        WEAK_CLASSIFIERS,
        "cifar100_single_resent/2020-12-02T15-21-48_700332_weight_decay_0_0001_linear_search_False/tb",
    ),
)
parser.add_argument(
    "--classifiers_indexes",
    type=str2intlist,
    default="[0,1,2,3,4,5,6,7,8,9]",
)
config = smart_parse_args(parser)

clebert_class = models.__dict__[config.model]
clebert = clebert_class(config)
clebert.to(config.device)
# optimizer = Adam(clebert.parameters(), lr=config.lr)
optimizer = optim.__dict__[config.optimizer](clebert.parameters(), lr=config.lr)
train_dataloader, eval_dataloader = prepare_data_loader(config)

"""loss function"""
loss_function_class = losses.__dict__[config.loss]
loss_function = loss_function_class(config)

if config.load_checkpoint is not None:
    if len(config.load_checkpoint) > 0:
        state_dict = torch.load(config.load_checkpoint)
        clebert.load_state_dict(state_dict)

best_total_loss = 1e10
if config.skip_training:
    epochs = 1
else:
    epochs = config.epochs

"""class that measures accuracy during training"""


class AccuracyMeter:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, output, ground_truth):
        target = ground_truth.argmax(dim=2)
        pred = output.argmax(dim=2)
        correct = pred.eq(target).sum().item()
        self.correct += correct

        # we have prediction for each expert
        self.total += target.shape[0] * target.shape[1]

    def get_accuracy(self):
        return self.correct / self.total * 100.0


class DataPreparator(object):
    def __init__(self, config):
        self.config = config
        if not self.config.use_static_files:
            self.clapool = models.CLAPOOL(config)

    def prepare_data(self, batch):
        if self.config.use_static_files:
            data = batch["cifar_env_response"]
        else:
            with torch.no_grad():
                data = self.clapool(batch["image"])
        """prepare data"""
        corrupted, masked_indexes, indexes = zeroout_experts(
            data,
            config.zeroout_prob,
            fixed_num=config.fixed_zero_exp_num,
        )
        corrupted = corrupted.to(config.device)
        gt = batch["cifar_env_response"].to(config.device)
        return corrupted, gt, indexes, masked_indexes


data_preparator = DataPreparator(config)
""" Main training loop """
for epoch in range(epochs):
    clebert.eval()
    eval_total_loss = 0
    val_accuracy_meter = AccuracyMeter()
    train_accuracy_meter = AccuracyMeter()
    """ Evaluate on the validation set """
    if not config.skip_validation:
        with torch.no_grad():
            with tqdm.tqdm(total=len(eval_dataloader)) as pbar:
                for batch in eval_dataloader:

                    """prepare data"""

                    (
                        corrupted,
                        gt,
                        indexes,
                        masked_indexes,
                    ) = data_preparator.prepare_data(batch)

                    # forward pass
                    output = clebert(corrupted, masked_indexes)

                    # extract masked outputs for accuracy calculation
                    masked_indexes = masked_indexes.to(config.device)
                    restored_embeddings = torch.masked_select(
                        output["restored_resp"], masked_indexes.bool().unsqueeze(-1)
                    ).view(gt.shape[0], len(indexes[0]), -1)
                    gt_restored = torch.masked_select(
                        gt, masked_indexes.bool().unsqueeze(-1)
                    ).view(gt.shape[0], len(indexes[0]), -1)

                    val_accuracy_meter.update(restored_embeddings, gt_restored)
                    val_accuracy = val_accuracy_meter.get_accuracy()
                    # compute loss
                    loss = loss_function(output, gt)
                    eval_total_loss += loss.item()
                    pbar.set_description(
                        f"epoch {epoch} / {config.epochs}:  total_loss:{eval_total_loss:.2f}"
                        f"  val_accuracy: {val_accuracy:.2f}"
                    )
                    pbar.update(1)
                val_accuracy = val_accuracy_meter.get_accuracy()
            print(f"eval_loss:{eval_total_loss}")
            if eval_total_loss < best_total_loss and not config.skip_training:
                best_path = os.path.join(config.output_dir, "best_net.pkl")
                torch.save(clebert.state_dict(), best_path)
                with open("best_net.info", "w") as fp:
                    json.dump({"epoch": epoch, "eval_total_loss": eval_total_loss}, fp)
                best_total_loss = eval_total_loss

    """ Train on the training set """
    if not config.skip_training:
        clebert.train()
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            train_total_loss = 0
            for ind, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                # corrupted, masked_indexes, indexes = zeroout_experts(
                #     batch["cifar_env_response"],
                #     config.zeroout_prob,
                #     fixed_num=config.fixed_zero_exp_num,
                # )

                (
                    corrupted,
                    gt,
                    indexes,
                    masked_indexes,
                ) = data_preparator.prepare_data(batch)
                corrupted = corrupted.to(config.device)

                output = clebert(corrupted, masked_indexes)
                loss = loss_function(output, gt)
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()

                masked_indexes = masked_indexes.to(config.device)
                restored_embeddings = torch.masked_select(
                    output["restored_resp"], masked_indexes.bool().unsqueeze(-1)
                ).view(gt.shape[0], len(indexes[0]), -1)
                gt_restored = torch.masked_select(
                    gt, masked_indexes.bool().unsqueeze(-1)
                ).view(gt.shape[0], len(indexes[0]), -1)

                train_accuracy_meter.update(restored_embeddings, gt_restored)
                train_accuracy = train_accuracy_meter.get_accuracy()

                pbar.update(1)
                pbar.set_description(
                    f"train_total_loss:{train_total_loss:.2f}"
                    f"  train_accuracy: {train_accuracy:.2f}"
                )

            train_accuracy = train_accuracy_meter.get_accuracy()
            write_wandb_dict(
                {
                    "train_accuracy": train_accuracy_meter.get_accuracy(),
                    "val_accuracy": val_accuracy_meter.get_accuracy(),
                    "train_total_loss": train_total_loss,
                    "val_total_loss": eval_total_loss,
                    "custom_step": epoch,
                },
                commit=True,
            )
