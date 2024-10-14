import numpy as np
import torch
import torch.nn as nn
from torch import concat, diag, logical_and, logical_or, tile

# def create_causal_mask(seq_len, device="cuda"):
#     causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
#     return causal_mask


def create_padding_mask(input_ids, device):
    padding_mask = (input_ids == 0).unsqueeze(1).unsqueeze(2)
    return padding_mask.to(device)


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_units, max_len, num_heads, num_layers, dropout_rate, device):
        super().__init__()
        self.max_len = max_len
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, hidden_units)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=1,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.future_mask_const = torch.triu(
            torch.ones(self.max_len, self.max_len) * float("-inf"), diagonal=1, device=self.device
        )
        self.seq_diag_const = ~diag(torch.ones(self.max_len, dtype=torch.bool, device=self.device))

    def forward(self, input_ids, pos_ids, neg_ids):
        input_ids = input_ids.to(self.device)
        pos_ids = pos_ids.to(self.device)
        neg_ids = neg_ids.to(self.device)

        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        item_embeddings = self.item_emb(input_ids) * np.sqrt(self.hidden_units)
        item_embeddings += self.pos_emb(position_ids)
        # mask = self.create_causal_mask(seq_len, self.device)
        padding_mask = self.create_padding_mask(input_ids, self.device)
        mask = self.merge_attn_masks(padding_mask)
        output = self.encoder(item_embeddings, mask)

        pos_embs = self.item_emb(torch.tensor(pos_ids, device=self.device))
        neg_embs = self.item_emb(torch.tensor(neg_ids, device=self.device))
        pos_logits = output * pos_embs
        neg_logits = output * neg_embs
        return pos_logits.sum(dim=-1), neg_logits.sum(dim=-1)

    def predict(self, input_ids):
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        item_embeddings = self.item_emb(input_ids) * np.sqrt(self.hidden_units)
        item_embeddings = item_embeddings + self.pos_emb(position_ids)
        mask = self.create_causal_mask(seq_len, self.device)
        output = self.encoder(item_embeddings, mask)
        return output

    def create_causal_mask(self, seq_len, device):
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * -1e9, diagonal=1)
        return causal_mask

    def create_padding_mask(self, input_ids, device):
        padding_mask = (input_ids == 0).unsqueeze(1).unsqueeze(2)
        return padding_mask.to(device)

    def merge_attn_masks(self, padding_mask):
        batch_size = padding_mask.shape[0]
        seq_len = padding_mask.shape[1]

        padding_mask_broadcast = ~padding_mask.bool().unsqueeze(1)
        future_masks = tile(self.future_mask_const[:seq_len, :seq_len], (batch_size, 1, 1))
        merged_masks = logical_or(padding_mask_broadcast, future_masks)
        # Always allow self-attention to prevent NaN loss
        # See: https://github.com/pytorch/pytorch/issues/41508
        diag_masks = tile(self.seq_diag_const[:seq_len, :seq_len], (batch_size, 1, 1))
        return logical_and(diag_masks, merged_masks)
