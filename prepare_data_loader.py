import sys

sys.path.append(
    "/media/Data1/projects/new/least_action/git/least_action/train_classifiers/cifar"
)

import os
from pathlib import Path

import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from local_config import WEAK_CLASSIFIERS


def list_files_in_folder(folder, pattern="*test_responses.npy"):
    all_nets = list(Path(folder).rglob(pattern))
    all_nets = [str(n) for n in all_nets]
    return sorted(all_nets)


def get_cifar_env_response_files2(
    classifiers_indexes, load_subnetworks, weak_classifier_folder
):
    # config = config
    # classifiers_indexes = config.cifar_classifier_indexes
    # self.classifier_responses_train = []
    # self.classifier_responses_test = []

    classifier_responses = []

    # device = "cpu"
    # if config.use_gpu:
    #     device = "cuda:0"

    if not load_subnetworks:
        loaded_networks = []
        test_resp_files = list_files_in_folder(
            weak_classifier_folder, "*test_repsonses*npy"
        )
        train_resp_files = list_files_in_folder(
            weak_classifier_folder, "*train_repsonses*npy"
        )
        if len(classifiers_indexes) == 0:
            classifiers_indexes = list(range(len(test_resp_files)))

        test_resp_files_used = []
        train_resp_files_used = []
        for i in classifiers_indexes:
            merged_resp = np.concatenate(
                (np.load(train_resp_files[i]), np.load(test_resp_files[i])), axis=0
            )
            test_resp_files_used.append(test_resp_files[i])
            train_resp_files_used.append(train_resp_files[i])
            classifier_responses.append(merged_resp)

    return classifier_responses


class IndexedDataset(data.Dataset):
    def __init__(self, args, cifar, index_correction=0):
        self.cifar = cifar
        self._args = args
        self.size = len(self.cifar)
        self.replicates = 1
        self.index_correction = index_correction
        cifar_100_weak_classifiers_path = os.path.join(
            WEAK_CLASSIFIERS,
            "cifar100_single_resent/2020-12-02T15-21-48_700332_weight_decay_0_0001_linear_search_False/tb",
        )
        # if self._args.semi_environment == "cifar_env_responder":
        all_responses = get_cifar_env_response_files2(
            list(range(10)), False, cifar_100_weak_classifiers_path
        )
        self.all_responses = np.stack(all_responses, axis=1)

    def __getitem__(self, index):
        image, label = self.cifar[index]
        # classifier_responses = None
        corrected_index = index + self.index_correction
        out_dict = {"image": image, "label": label, "index": corrected_index}

        # if self._args.semi_environment == "cifar_env_responder":
        # resp = self.all_responses[corrected_index]
        resp = self.all_responses[0]
        out_dict["cifar_env_response"] = resp

        return out_dict

    def __len__(self):
        return self.size * self.replicates


def prepare_data_loader(config):
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    dataset_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=trans
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=trans
    )

    # dataset = IndexedDataset(None,)

    dataset_train = IndexedDataset(config, dataset_train)
    dataset_test = IndexedDataset(config, dataset_test, len(dataset_train))
    # classifier_responses = get_cifar_env_response_files2(list(range(10)), False, cifar_100_weak_classifiers_path)
    # classifier_responses = np.stack(classifier_responses,axis=1)
    # IndexedData
    train_dataloader = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    eval_dataloader = DataLoader(
        dataset_test, batch_size=config.batch_size, num_workers=config.num_workers
    )
    return train_dataloader, eval_dataloader
