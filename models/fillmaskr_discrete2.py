from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel


class FillMaskRandDiscrete2(nn.Module):
    """
    Bert to predict classifiers responses in discrete form.

    First convert classifiers responses to indexes.
    Index corresponds to the max response (predicted image class).
    """

    def __init__(self, config):
        """
        Initialize all the necessary modules of the neural network.

        Parameters
        -----
        config : namespace
              program attributes
        """
        self.config = config
        self.device = config.device
        super(FillMaskRandDiscrete2, self).__init__()

        if self.config.use_pretrained_bert:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            print("using pretrained bert")
        else:
            bert_config = AutoConfig.from_pretrained("bert-base-uncased")
            self.bert = BertModel(bert_config)
            print("using randomly initialized bert")

        words_indexes = torch.Tensor([list(range(10))]).long().to(self.device)
        self.bert.to(self.device)
        self.word_positional_embeddings = self.bert.embeddings.position_embeddings(
            words_indexes
        ).detach()

        self.vector2response = nn.Linear(768, 100)

        word_type = torch.Tensor([[1 for _ in range(10)]]).long().to(self.device)
        self.type_embedding = self.bert.embeddings.token_type_embeddings(
            word_type
        ).detach()

    def forward(self, corrupted_responses, indexes) -> Dict[str, torch.Tensor]:
        """
        Make prediction.

        Parameters:
            corrupted_responses: response table with randomly zeroed classifeirs' responses
            indexes: indexes of the zeroed responses
        """
        # compute the indexes of the max responses
        words = corrupted_responses.argmax(dim=2)

        # use word2vec in bert
        words_embeddings = self.bert.embeddings.word_embeddings(words)

        type_embeddings = self.type_embedding
        if indexes is not None:
            tmp = indexes.long().to(self.device)
            type_embeddings = self.bert.embeddings.token_type_embeddings(tmp).detach()
        final_embeddings = (
            words_embeddings + self.word_positional_embeddings + type_embeddings
        )
        final_embeddings = self.bert.embeddings.LayerNorm(final_embeddings)
        vectors = self.bert.encoder(final_embeddings)["last_hidden_state"]
        responses = self.vector2response(vectors)

        # responses = self.bert.encoder(corrupted_responses)['last_hidden_state']
        return {"restored_resp": responses}
