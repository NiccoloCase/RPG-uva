from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _ensure_third_party_on_path() -> None:
    """Expose the vendored `genrec` package without editing the submodule."""
    repo_root = Path(__file__).resolve().parents[2]
    third_party_root = repo_root / "third_party"
    if str(third_party_root) not in sys.path:
        sys.path.insert(0, str(third_party_root))


_ensure_third_party_on_path()

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class SASRecBlock(torch.nn.Module):
    """Single SASRec self-attention block."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=attn_dropout_prob,
        )
        self.attention_layer_norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_layer_norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.linear1 = torch.nn.Linear(hidden_size, ffn_hidden_size)
        self.linear2 = torch.nn.Linear(ffn_hidden_size, hidden_size)
        self.activation = self._get_activation(hidden_act)
        self.attention_dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.ffn_dropout = torch.nn.Dropout(hidden_dropout_prob)

    @staticmethod
    def _get_activation(hidden_act: str) -> torch.nn.Module:
        activation_map = {
            "gelu": torch.nn.GELU(),
            "relu": torch.nn.ReLU(),
            "swish": torch.nn.SiLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
        }
        if hidden_act not in activation_map:
            raise ValueError(f"Unsupported SASRec activation: {hidden_act}")
        return activation_map[hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.attention_layer_norm(hidden_states)
        attn_output, _ = self.self_attention(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.attention_dropout(attn_output)

        ffn_input = self.ffn_layer_norm(hidden_states)
        ffn_output = self.linear2(self.ffn_dropout(self.activation(self.linear1(ffn_input))))
        hidden_states = hidden_states + self.ffn_dropout(ffn_output)

        # Keep padded positions inactive across blocks.
        hidden_states = hidden_states.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return hidden_states


class SASRec(AbstractModel):
    """Repo-owned SASRec scaffold.

    The canonical architecture to clone is:
    - item embedding table over raw item IDs,
    - learned positional embeddings up to `max_item_seq_len`,
    - stacked causal self-attention blocks,
    - point-wise feed-forward sublayers with residual connections,
    - final hidden state at the last non-padding position as the user state,
    - next-item scoring against candidate item embeddings.

    The original paper trains with a binary objective over positive and
    sampled-negative next items. Some reproductions instead use a full-softmax
    cross-entropy loss. Pick one and keep the tokenizer/model interfaces
    consistent with the trainer you plan to use.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super().__init__(config, dataset, tokenizer)
        self.n_items = dataset.n_items
        self.max_seq_len = tokenizer.max_token_seq_len
        self.hidden_size = config["hidden_size"]
        self.num_blocks = config.get("n_layers", config.get("num_blocks", 2))
        self.num_attention_heads = config.get("n_heads", config.get("num_attention_heads", 2))
        self.ffn_hidden_size = config.get("inner_size", config.get("ffn_hidden_size", 256))
        self.hidden_dropout_prob = config.get("hidden_dropout_prob", config.get("dropout_prob", 0.5))
        self.attn_dropout_prob = config.get("attn_dropout_prob", self.hidden_dropout_prob)
        self.hidden_act = config.get("hidden_act", "gelu")
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-12)
        self.initializer_range = config.get("initializer_range", 0.02)

        self.item_embedding = torch.nn.Embedding(
            self.n_items,
            self.hidden_size,
            padding_idx=0,
        )
        self.positional_embedding = torch.nn.Embedding(
            self.max_seq_len,
            self.hidden_size,
        )
        self.input_layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.input_dropout = torch.nn.Dropout(self.hidden_dropout_prob)
        self.blocks = torch.nn.ModuleList(
            [
                SASRecBlock(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    ffn_hidden_size=self.ffn_hidden_size,
                    hidden_dropout_prob=self.hidden_dropout_prob,
                    attn_dropout_prob=self.attn_dropout_prob,
                    hidden_act=self.hidden_act,
                    layer_norm_eps=self.layer_norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.final_layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        ).unsqueeze(0).expand_as(input_ids)

        hidden_states = self.item_embedding(input_ids) * (self.hidden_size ** 0.5)
        hidden_states = hidden_states + self.positional_embedding(position_ids)
        hidden_states = self.input_layer_norm(hidden_states)
        hidden_states = self.input_dropout(hidden_states)
        hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(
            torch.ones(input_ids.size(1), input_ids.size(1), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = ~attention_mask

        for block in self.blocks:
            hidden_states = block(hidden_states, causal_mask, key_padding_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = hidden_states.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        seq_indices = (attention_mask.sum(dim=1) - 1).clamp_min(0)
        seq_output = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            seq_indices,
        ]
        return seq_output

    def forward(self, batch: dict, return_loss: bool = True):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()
        seq_output = self._encode(input_ids, attention_mask)
        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        outputs = SimpleNamespace(seq_output=seq_output, logits=logits)

        if return_loss:
            labels = batch["labels"]
            if labels.dim() == 2:
                labels = labels.squeeze(-1)
            outputs.loss = self.loss_fct(logits, labels)

        return outputs

    def generate(self, batch: dict, n_return_sequences: int = 1):
        outputs = self.forward(batch, return_loss=False)
        scores = outputs.logits.clone()
        scores.scatter_(1, batch["input_ids"], float("-inf"))
        scores[:, 0] = float("-inf")
        topk_items = torch.topk(scores, k=n_return_sequences, dim=-1).indices
        return topk_items.unsqueeze(-1)
