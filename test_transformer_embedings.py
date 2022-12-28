import os
import sys
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from local_config import WEAK_CLASSIFIERS
from datasets_cifar import list_files_in_folder

sys.path.append(
     "/media/Data1/projects/new/least_action/git/least_action/train_classifiers/cifar"
)

def get_cifar_env_response_files2(classifiers_indexes, load_subnetworks, weak_classifier_folder):
    # config = config
    # classifiers_indexes = config.cifar_classifier_indexes
    # self.classifier_responses_train = []
    # self.classifier_responses_test = []

    classifier_responses = []

    # device = "cpu"
    # if config.use_gpu:
    #     device = "cuda:0"

    if not load_subnetworks:
        # loaded_networks = []
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


cifar_100_weak_classifiers_path = os.path.join(
    WEAK_CLASSIFIERS,
    "cifar100_single_resent/2020-12-02T15-21-48_700332_weight_decay_0_0001_linear_search_False/tb",
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased")

text = "Replace me by any text you'd like [MASK]."

encoded_input = tokenizer(text, return_tensors="pt")

output = model(**encoded_input)

words_type = torch.Tensor([[0, 1, 1]]).long()

words = torch.Tensor([list(range(words_type.shape[1]))]).long()

words_indexes = torch.Tensor([list(range(words_type.shape[1]))]).long()

attention_mask = torch.Tensor([1 for i in range(words_type.shape[1])]).long()

full_embeding = model.embeddings(input_ids=words, token_type_ids=words_type)

a = model.embeddings.word_embeddings(words)
b = model.embeddings.position_embeddings(words_indexes)
c = model.embeddings.token_type_embeddings(words_type)
d = model.embeddings.LayerNorm(a + b + c)

d == full_embeding
print(d)
