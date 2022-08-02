# type: ignore
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class IndexFiller(nn.Module):
    def __init__(self, config):

        self.config = config
        self.device = config.device
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        words_indexes = torch.Tensor([list(range(10))]).long().to(self.device)
        self.bert.to(self.device)
        self.word_positional_embeddings = self.bert.embeddings.position_embeddings(
            words_indexes
        ).detach()
        self.response2embedding = nn.Linear(100, 768).to(self.device)
        self.vector2response = nn.Linear(768, 100)
        word_type = torch.Tensor([[1 for _ in range(10)]]).long().to(self.device)
        self.type_embedding = self.bert.embeddings.token_type_embeddings(
            word_type
        ).detach()

    def forward(self, corrupted_responses, indexes):

        word_indicies = corrupted_responses.argmax(dim=2)
        word_indicies[indexes.bool()] = 1001
        tmp = indexes.long().to(self.device)

        final_embeddings = self.bert.embeddings(
            input_ids=word_indicies, token_type_ids=tmp
        )

        vectors = self.bert.encoder(final_embeddings)["last_hidden_state"]
        responses = self.vector2response(vectors)

        return {"restored_resp": responses}
