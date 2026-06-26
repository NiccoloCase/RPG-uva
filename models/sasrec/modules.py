from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": F.silu,
}


class SASRecSelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{args.hidden_size} vs {args.num_attention_heads}"
            )

        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = args.hidden_size // args.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def _manual_attention(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        return torch.matmul(attention_probs, value_layer)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query_layer = self._transpose_for_scores(self.query(input_tensor))
        key_layer = self._transpose_for_scores(self.key(input_tensor))
        value_layer = self._transpose_for_scores(self.value(input_tensor))

        if hasattr(F, "scaled_dot_product_attention"):
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            context_layer = self._manual_attention(query_layer, key_layer, value_layer, attention_mask)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.reshape(*context_layer.shape[:-2], self.all_head_size)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class SASRecIntermediate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.intermediate_act_fn = ACT2FN[args.hidden_act] if isinstance(args.hidden_act, str) else args.hidden_act
        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class SASRecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SASRecSelfAttention(args)
        self.intermediate = SASRecIntermediate(args)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        return self.intermediate(attention_output)


class SASRecEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = nn.ModuleList(
            SASRecLayer(args) for _ in range(args.num_hidden_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ) -> list[torch.Tensor]:
        all_encoder_layers: list[torch.Tensor] = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
