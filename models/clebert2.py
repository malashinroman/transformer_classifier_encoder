import torch
import torch.nn as nn
from transformers import BertModel


class CleBert3(nn.Module):
    def __init__(self, config):
        self.config = config
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.device = config.device
        self.bert.to(self.device)
        words_indexes = torch.Tensor([list(range(10))]).long().to(self.device)
        self.response2embedding = nn.Linear(100, 768).to(self.device)
        self.vector2response = nn.Linear(768, 100)

        # create positional embedding from 1 to 10
        self.word_positional_embeddings = self.bert.embeddings.position_embeddings(
            words_indexes
        ).detach()

        # types of words (similar)
        word_type = torch.ones(0).long().to(self.device)
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
