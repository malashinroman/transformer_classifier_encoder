import json
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import six
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(".")
import datasets

# from datasets.imagenet_rec import ImageNetREC
from local_config import IMAGENET_PATH, WEAK_CLASSIFIERS


def list_files_in_folder(folder, pattern="*test_responses.npy"):
    all_nets = list(Path(folder).rglob(pattern))
    all_nets = [str(n) for n in all_nets]
    return sorted(all_nets)


def read_json(filename):
    def _convert_from_unicode(data):
        new_data = dict()
        for name, value in six.iteritems(data):
            if isinstance(name, six.string_types):
                name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore")
            if isinstance(value, six.string_types):
                value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
            if isinstance(value, dict):
                value = _convert_from_unicode(value)
            new_data[name] = value
        return new_data

    output_dict = None
    with open(filename, "r") as f:
        lines = f.readlines()
        try:
            output_dict = json.loads("".join(lines), encoding="utf-8")
        except:
            raise ValueError("Could not read %s. %s" % (filename, sys.exc_info()[1]))
        output_dict = _convert_from_unicode(output_dict)
    return output_dict


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

        # else:
        #     nets = list_files_in_folder(weak_classifier_folder, '*.pkl')
        #     jsons = list_files_in_folder(weak_classifier_folder, '*.json')
        #     net_models = [str(read_json(json)[b'model'])[2:-1] for json in jsons]
        #     architectures = models
        #     # self.loaded_networks = [architectures[net_model](config, [3,32,32]) for net_model in net_models]
        #     self.loaded_networks = torch.nn.ModuleList()

        #     for i in classifiers_indexes:
        #         data = torch.load(nets[i])
        #         model_name = net_models[i]
        #         input_shape = self.config.dataset_image_shape
        #         if self.config.patch_size > 0:
        #             input_shape = (self.config.dataset_image_shape[0], self.config.patch_size, self.config.patch_size)
        #         loaded_network = architectures[model_name](config, input_shape)
        #         # if 'total_ops' in data['state_dict']:
        #             # from thop import profile
        #             # profile(loaded_network, inputs=({'image': torch.zeros((1,) +input_shape).cuda()},))
        #         load_state_dict_into_module(data['state_dict'], loaded_network)
        #         # loaded_network.load_state_dict(state_dict=data['state_dict'])
        #         loaded_network = loaded_network.to(self.device)
        #         self.loaded_networks.append(loaded_network)

    return classifier_responses


class IndexedDataset(data.Dataset):
    """class to add unique index to each image.
    Adds classifiers_response to the image if provided.
    """

    def __init__(self, args, cifar, index_correction=0):
        self.cifar = cifar
        self._args = args
        self.size = len(self.cifar)
        self.replicates = 1
        self.index_correction = index_correction
        cifar_100_weak_classifiers_path = args.weak_classifier_folder
        if self._args.use_static_files:
            all_responses = get_cifar_env_response_files2(
                self._args.classifiers_indexes, False, cifar_100_weak_classifiers_path
            )
            self.all_responses = np.stack(all_responses, axis=1)

    def __getitem__(self, index):
        image, label = self.cifar[index]
        corrected_index = index + self.index_correction
        out_dict = {"image": image, "label": label, "index": corrected_index}

        if self._args.use_static_files:
            resp = self.all_responses[corrected_index]
            # resp = self.all_responses[0]
            out_dict["cifar_env_response"] = resp
        return out_dict

    def __len__(self):
        return self.size * self.replicates


def prepare_data_loader(config):
    """return dataloaders for training"""
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    if config.dataset == "cifar100":
        # FIXME: SHOULD BE CIFAR100-here??
        dataset_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=trans
        )
        dataset_test = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=trans
        )
    elif config.dataset == "imagenet":
        dataset_train = datasets.IMAGENET_DATASET(
            root=IMAGENET_PATH, image_size=32, transform=trans, split="train"
        )
        dataset_test = datasets.IMAGENET_DATASET(
            root=IMAGENET_PATH, image_size=32, transform=trans, split="val"
        )

        # need to have tar files for pytorch interface
        # dataset_train = torchvision.datasets.ImageNet(
        #     root=IMAGENET_PATH, split="train", target_transform=trans
        # )
        # dataset_test = torchvision.datasets.ImageNet(
        #     root=IMAGENET_PATH, split="val", target_transform=trans
        # )

    else:
        raise ValueError("unknown dataset_type")

    # dataset = IndexedDataset(None,)
    indices_train = list(range(len(dataset_train)))
    indices_test = list(range(len(dataset_test)))

    if config.train_set_size > 0:
        indices_train = indices_train[0 : config.train_set_size]

    if config.test_set_size > 0:
        indices_test = indices_test[0 : config.test_set_size]

    dataset_train = data.Subset(dataset_train, indices_train)
    dataset_test = data.Subset(dataset_test, indices_test)

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
