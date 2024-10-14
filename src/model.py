import torch
import torch.nn as nn


def create_causal_mask(seq_len):
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * -1e9, diagonal=1)
    return causal_mask


class SASRec(nn.Module):
    def __init__(
        self, num_items, hidden_units, max_len, num_heads, num_layers, dropout_rate
    ):
        super().__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, hidden_units)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=1,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.causal_mask = create_causal_mask(self.max_len)

    def forward(self, input_ids, pos_ids, neg_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        item_embeddings = self.item_emb(input_ids) + self.pos_emb(position_ids)
        mask = create_causal_mask(seq_len)
        output = self.encoder(item_embeddings, mask)

        pos_embs = self.item_emb(torch.LongTensor(pos_ids))
        neg_embs = self.item_emb(torch.LongTensor(neg_ids))
        pos_logits = output * pos_embs
        neg_logits = output * neg_embs
        return pos_logits.sum(dim=-1), neg_logits.sum(dim=-1)

    def predict(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        item_embeddings = self.item_emb(input_ids) + self.pos_emb(position_ids)
        mask = create_causal_mask(seq_len)
        output = self.encoder(item_embeddings, mask)
        return output
