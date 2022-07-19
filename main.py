import argparse
import json
import os
import sys

import numpy as np
import torch
import tqdm
from torch.optim import Adam

import losses
import models
from prepare_data_loader import prepare_data_loader
from script_manager.func.add_needed_args import smart_parse_args
from utils import zeroout_experts

sys.path.append(".")


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--load_checkpoint", default=None, type=str)
parser.add_argument("--losses", default="AE_MSE_LOSS", type=str)
parser.add_argument("--model", default="Clebert", type=str)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--skip_training", default=0, type=int)
parser.add_argument("--skip_validation", default=0, type=int)
parser.add_argument("--loss", default="AE_MSE_LOSS", type=str)
parser.add_argument("--zeroout_prob", default=0.15, type=float)
parser.add_argument("--lr", default=1e-3, type=float)

config = smart_parse_args(parser)

clebert_class = models.__dict__[config.model]
clebert = clebert_class(config)
clebert.to(config.device)
optimizer = Adam(clebert.parameters(), lr=config.lr)

train_dataloader, eval_dataloader = prepare_data_loader(config)

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


for epoch in range(epochs):
    clebert.eval()
    eval_total_loss = 0
    if not config.skip_validation:
        with torch.no_grad():
            with tqdm.tqdm(total=len(eval_dataloader)) as pbar:
                for batch in eval_dataloader:
                    corrupted, indexes = zeroout_experts(
                        batch["cifar_env_response"], config.zeroout_prob
                    )
                    corrupted = corrupted.to(config.device)
                    output = clebert(corrupted, indexes)

                    loss = loss_function(batch, output)
                    eval_total_loss += loss.item()
                    pbar.set_description(
                        f"epoch {epoch} / {config.epochs}:  total_loss:{eval_total_loss:.2f}"
                    )
                    pbar.update(1)
            print(f"eval_loss:{eval_total_loss}")
            if eval_total_loss < best_total_loss and not config.skip_training:
                torch.save(clebert.state_dict(), "best_net.pkl")
                with open("best_net.info", "w") as fp:
                    json.dump({"epoch": epoch, "eval_total_loss": eval_total_loss}, fp)
    if not config.skip_training:
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            clebert.train()
            train_total_loss = 0
            for ind, batch in enumerate(train_dataloader):
                # outputs = model(**batch)
                # optimizer.zero_grad()
                optimizer.zero_grad()
                corrupted, indexes = zeroout_experts(
                    batch["cifar_env_response"], config.zeroout_prob
                )
                # corrupted = torch.zeros_like(corrupted)
                # corrupted = torch.ones([config.batch_size, 10, 768]).cuda() * 100
                # batch["cifar_env_response"] = corrupted

                corrupted = corrupted.to(config.device)
                output = clebert(corrupted, indexes)
                loss = loss_function(batch, output)
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                train_total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f"train_total_loss:{train_total_loss:.2f}")

            print(f"train_total_loss:{train_total_loss:.3f}")
