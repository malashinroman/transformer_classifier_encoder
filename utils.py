import torch
import random

def zeroout_experts(tensor_, prob):
    if prob == 0.0:
        return tensor_
    tensor = tensor_.clone()
    batch_size = tensor.shape[0]
    expert_num = tensor.shape[1]
    indexes = list(range(expert_num))
    all_experts_to_destroy = []
    for i in range(batch_size):
        expert_to_destroy = [i for i in indexes if random.random() < prob]
        ttt = [1 if e in expert_to_destroy else 0 for e in range(expert_num)]
        all_experts_to_destroy.append(torch.Tensor(ttt))
        # FIXME: There can be all or  nothing cases.
        # Is it bad o not bad (maybe we can learn 'mean' distribution)?
        for k in expert_to_destroy:
            tensor[i][k] = torch.zeros_like(tensor[i][k])

    return tensor, torch.stack(all_experts_to_destroy, dim=0)


