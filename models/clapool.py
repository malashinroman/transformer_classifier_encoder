import torch
import torch.nn as nn

import cnn_models
from prepare_data_loader import list_files_in_folder, read_json


def copy_weight_part(own_state, name, param):
    if own_state[name].shape == param.shape:
        own_state[name].resize_as_(param)
        own_state[name].copy_(param)
    else:
        print(
            f'Skipping weight "{name}" copying. {own_state[name].shape} (dst) !={param.shape} (src)'
        )


def load_state_dict_into_module(state_dict, module, strict=False, ignore_prefix=[]):
    own_state = module.state_dict()
    not_loaded = set(state_dict.keys()) - set(own_state.keys())
    print(
        f"{len(state_dict.keys())}/{len(state_dict.keys()) - len(not_loaded)} weight elements can be loaded"
    )
    for name, param in state_dict.items():
        for pr in ignore_prefix:
            name = name.replace(pr, "")

        all_names_with_prefixes = [name] + [p + name for p in ignore_prefix]
        satisfies = [p in own_state for p in all_names_with_prefixes]
        satisf_indexes = [i for i, x in enumerate(satisfies) if x]
        assert len(satisf_indexes) < 2
        if len(satisf_indexes) > 0:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state_name = all_names_with_prefixes[satisf_indexes[0]]
                copy_weight_part(own_state, own_state_name, param)
            except Exception:
                raise RuntimeError(
                    "While copying the parameter named {}., "
                    "whose dimensions in the model are {} and "
                    "whose dimensions in teh checkpoint are {}".format(
                        name, own_state[name].size(), param.size()
                    )
                )
        elif strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            raise KeyError('unexpected key "{}" in state_dict'.format(missing))
        else:
            print(f"Not loaded: {name}")


class ClaPool(nn.Module):
    """Interface to the cnn networks to get the responses online"""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        super(ClaPool, self).__init__()

        # load cnn_networks weights
        # it was nessesary to resave nets state_dict only in *.pkl2
        nets = list_files_in_folder(config.weak_classifier_folder, "*.pkl2")
        jsons = list_files_in_folder(config.weak_classifier_folder, "*.json")
        net_models = [str(read_json(json)[b"model"])[2:-1] for json in jsons]
        architectures = cnn_models.__dict__
        # self.loaded_networks = [architectures[net_model](config, [3,32,32]) for net_model in net_models]
        self.loaded_networks = torch.nn.ModuleList()

        for i in self.config.classifiers_indexes:
            model_name = net_models[i]
            data = torch.load(nets[i])
            # input_shape = self.config.dataset_image_shape

            # hardcoded for now
            config.dataset_image_shape = [3, 32, 32]
            config.n_classes = 100
            config.torch_model = "resnet18"
            input_shape = config.dataset_image_shape

            loaded_network = architectures[model_name](config, input_shape)

            load_state_dict_into_module(data, loaded_network)
            loaded_network = loaded_network.to(self.device)
            self.loaded_networks.append(loaded_network)

        for net in self.loaded_networks:
            for p in net.parameters():
                p.requires_grad = False

    def forward(self, images):
        responses = []
        for net in self.loaded_networks:
            net_input = {"image": images.to(self.device)}
            responses.append(net(net_input)["prediction"])

        responses = torch.stack(responses, dim=1)
        # responses = self.bert.encoder(corrupted_responses)['last_hidden_state']
        return responses
