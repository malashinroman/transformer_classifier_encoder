import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertModel

# from transformers.models.auto.modeling_auto import AutoModelForTokenClassification


class TokenClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER"
        )
        self.device = config.device
        self.model.to(self.device)
        words_indexes = torch.Tensor([list(range(10))]).long().to(self.device)
        self.response2embedding = nn.Linear(100, 768).to(self.device)
        self.vector2response = nn.Linear(768, 100)

        # create positional embedding from 1 to 10
        self.word_positional_embeddings = (
            self.model.bert.embeddings.position_embeddings(words_indexes).detach()
        )

        # types of words (similar)
        word_type = torch.ones(0).long().to(self.device)
        self.type_embedding = self.model.bert.embeddings.token_type_embeddings(
            word_type
        ).detach()

    def forward(self, corrupted_responses, indexes):
        embeddings = self.response2embedding(corrupted_responses.to(self.device))
        type_embeddings = self.type_embedding
        if indexes is not None:
            tmp = indexes.long().to(self.device)
            type_embeddings = self.model.bert.embeddings.token_type_embeddings(
                tmp
            ).detach()
        final_embeddings = (
            embeddings + self.word_positional_embeddings + type_embeddings
        )
        final_embeddings = self.model.bert.embeddings.LayerNorm(final_embeddings)
        vectors = self.model.bert.encoder(final_embeddings)["last_hidden_state"]
        responses = self.vector2response(vectors)

        # responses = self.model.encoder(corrupted_responses)['last_hidden_state']
        return {"restored_resp": responses}
