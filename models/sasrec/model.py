from __future__ import annotations

import torch
import torch.nn as nn

from .modules import SASRecEncoder


class SASRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = SASRecEncoder(args)
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.layer_norm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    def _build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(
            torch.ones(attn_shape, device=input_ids.device),
            diagonal=1,
        )
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=self.item_embeddings.weight.dtype)
        return (1.0 - extended_attention_mask) * -10000.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        extended_attention_mask = self._build_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        return encoded_layers[-1]

    def init_weights(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
