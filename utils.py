import random

import numpy as np
import torch


def zeroout_experts(tensor_: torch.Tensor, prob: float, fixed_num: int = 0):
    if prob == 0.0:
        return tensor_
    tensor = tensor_.clone()
    batch_size = tensor.shape[0]
    expert_num = tensor.shape[1]
    indexes = list(range(expert_num))
    all_experts_to_destroy = []
    for i in range(batch_size):
        if fixed_num <= 0:
            expert_to_destroy = [i for i in indexes if random.random() < prob]
        else:
            # __import__("pudb").set_trace()
            expert_to_destroy = random.sample(range(0, expert_num), fixed_num)

        ttt = [1 if e in expert_to_destroy else 0 for e in range(expert_num)]
        all_experts_to_destroy.append(torch.Tensor(ttt))

        # FIXME: There can be all or  nothing cases.
        # Is it bad o not bad (maybe we can learn 'mean' distribution)?
        for k in expert_to_destroy:
            tensor[i][k] = torch.zeros_like(tensor[i][k])

    return tensor, torch.stack(all_experts_to_destroy, dim=0)
