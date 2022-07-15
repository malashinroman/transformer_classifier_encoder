import os
import sys

sys.path.append(".")
import argparse
import json
import random

import numpy as np

# sys.path.append('/media/Data1/projects/new/least_action/git/least_action/train_classifiers/cifar')
import torch
import tqdm
from datasets import load_metric
from torch.optim import Adam, AdamW
from transformer_playground.transformer_encoder.prepare_data_loader import (
    prepare_data_loader,
)
from transformer_playground.transformer_encoder.utils import zeroout_experts
from transformers import BertModel, BertTokenizer, get_scheduler

import models
import wandb
from losses import AutoencoderLoss

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--skip_validation", default=0, type=int)
parser.add_argument("--skip_training", default=1, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--load_checkpoint", default="best_net500.pkl", type=str)
parser.add_argument("--model", default="Clebert", type=str)

config = parser.parse_args()

# clebert = CleBert(config)
clebert_class = models.__dict__[config.model]
clebert = clebert_class(config)
clebert.to(config.device)
optimizer = Adam(clebert.parameters(), lr=5e-3)

# lr_scheduler = get_scheduler(
#     name="linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=config.epochs,
# )

train_dataloader, eval_dataloader = prepare_data_loader(config)
ae_loss_func = AutoencoderLoss(config)

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
                        batch["cifar_env_response"], 0.6
                    )
                    output = clebert(corrupted, indexes)

                    loss = ae_loss_func(batch, output)
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
            clebert.eval()
            train_total_loss = 0
            for ind, batch in enumerate(train_dataloader):
                # outputs = model(**batch)
                # optimizer.zero_grad()
                optimizer.zero_grad()
                corrupted, indexes = zeroout_experts(batch["cifar_env_response"], 0.6)
                # corrupted = torch.zeros_like(corrupted)
                # corrupted = torch.ones([config.batch_size, 10, 768]).cuda() * 100
                # batch["cifar_env_response"] = corrupted
                output = clebert(corrupted, indexes)
                # if (ind > 100):
                #     break
                loss = ae_loss_func(batch, output)
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                train_total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f"train_total_loss:{train_total_loss:.2f}")
                # progress_bar.update(1)

            print(f"train_total_loss:{train_total_loss:.3f}")

# metric = load_metric("accuracy")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)
#
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])
#
# metric_estimation = metric.compute()
# print(metric_estimation)
#
# decoded_response = clebert(eval_dataloader)

# a = classifier_responses[0]
# words_indexes = torch.Tensor([list(range(10))]).long()
# b = model.embeddings.position_embeddings(words_indexes)
