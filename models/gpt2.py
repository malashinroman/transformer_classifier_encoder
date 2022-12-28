import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel, GPT2Model


class FillMaskGPT2(nn.Module):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        super(FillMaskGPT2, self).__init__()
        if self.config.hugging_face_model is not None:
            hugging_face_model = self.config.hugging_face_model
        else:
            hugging_face_model = "gpt2"

        self.gpt2 = GPT2Model.from_pretrained(hugging_face_model)
        # bert_config = AutoConfig.from_pretrained("bert-base-uncased")
        # self.bert = BertModel(bert_config)

        words_indexes = torch.Tensor([list(range(10))]).long().to(self.device)
        self.gpt2.to(self.device)
        self.word_positional_embeddings = self.gpt2.wpe(
            words_indexes
        ).detach()
        embed_size = self.gpt2.ln_f.normalized_shape[0]
        self.response2embedding = nn.Linear(100, embed_size).to(self.device)
        self.vector2response = nn.Linear(embed_size, 100)
        word_type = torch.Tensor([[1 for _ in range(10)]]).long().to(self.device)
        self.type_embedding = self.gpt2.wte(
            word_type
        ).detach()

    def forward(self, corrupted_responses, indexes):
        embeddings = self.response2embedding(corrupted_responses.to(self.device))
        type_embeddings = self.type_embedding
        if indexes is not None:
            tmp = indexes.long().to(self.device)
            type_embeddings = self.gpt2.wte(tmp).detach()

        batch_size = embeddings.shape[0]
        final_embeddings = (
            embeddings
            + self.word_positional_embeddings.repeat([batch_size, 1, 1])
            + type_embeddings
        )
        x = final_embeddings
        for i in range(len(self.gpt2.h)):
            x = self.gpt2.h[i](x)[0]

        # vectors = self.gpt2.encoder(x)["last_hidden_state"]
        responses = self.vector2response(x)

        # responses = self.bert.encoder(corrupted_responses)['last_hidden_state']
        return {"restored_resp": responses}
