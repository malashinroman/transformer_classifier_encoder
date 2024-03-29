# type: ignore
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class FillMask(nn.Module):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        super(FillMask, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        number_of_classifiers = len(config.classifiers_indexes)
        words_indexes = (
            torch.Tensor([list(range(number_of_classifiers))]).long().to(self.device)
        )
        self.bert.to(self.device)
        self.word_positional_embeddings = self.bert.embeddings.position_embeddings(
            words_indexes
        ).detach()
        self.response2embedding = nn.Linear(100, 768).to(self.device)
        self.vector2response = nn.Linear(768, 100)
        word_type = (
            torch.Tensor([[1 for _ in range(number_of_classifiers)]])
            .long()
            .to(self.device)
        )
        self.type_embedding = self.bert.embeddings.token_type_embeddings(
            word_type
        ).detach()

    def forward(self, corrupted_responses, indexes):
        embeddings = self.response2embedding(corrupted_responses.to(self.device))
        type_embeddings = self.type_embedding
        if indexes is not None:
            tmp = indexes.long().to(self.device)
            type_embeddings = self.bert.embeddings.token_type_embeddings(tmp).detach()
        final_embeddings = (
            embeddings + self.word_positional_embeddings + type_embeddings
        )
        final_embeddings = self.bert.embeddings.LayerNorm(final_embeddings)
        vectors = self.bert.encoder(final_embeddings)["last_hidden_state"]
        responses = self.vector2response(vectors)

        # responses = self.bert.encoder(corrupted_responses)['last_hidden_state']
        return {"restored_resp": responses}
