from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.models.RPG.model import RPG
from genrec.tokenizer import AbstractTokenizer


class Denoiser(nn.Module):
    """Non-causal masked semantic-ID denoiser.

    Target positions attend to each other and cross-attend to the encoded user
    history. Masked positions use learned per-digit mask embeddings; visible
    positions use the shared semantic-token embedding table.
    """

    def __init__(
        self,
        n_digit: int,
        n_embd: int,
        vocab_size: int,
        mask_token_id: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        do_norm_and_scale: bool,
    ):
        super().__init__()
        self.n_digit = n_digit
        self.mask_token_id = mask_token_id
        self.target_embeddings = nn.Embedding(vocab_size, n_embd)
        self.mask_embeddings = nn.Embedding(n_digit, n_embd)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(n_embd),
        )
        self.do_norm_and_scale = do_norm_and_scale

    def init_target_embeddings(self, gpt2_wte_weights: nn.Parameter) -> None:
        self.target_embeddings.weight = gpt2_wte_weights
        with torch.no_grad():
            self.mask_embeddings.weight.normal_(0, 0.02)

    def forward(
        self,
        target_tokens: torch.Tensor,
        user_history: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        is_masked = target_tokens == self.mask_token_id
        safe_tokens = target_tokens.masked_fill(is_masked, 0)

        token_embs = self.target_embeddings(safe_tokens)
        mask_pos = torch.arange(self.n_digit, device=target_tokens.device)
        mask_embs = self.mask_embeddings(mask_pos).unsqueeze(0).expand(target_tokens.size(0), -1, -1)

        if self.do_norm_and_scale:
            token_embs = F.normalize(token_embs, dim=-1, eps=1e-8)
            mask_embs = F.normalize(mask_embs, dim=-1, eps=1e-8)

        tgt = torch.where(is_masked.unsqueeze(-1), mask_embs, token_embs)
        return self.transformer(
            tgt=tgt,
            memory=user_history,
            memory_key_padding_mask=memory_padding_mask,
        )


class DRPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super().__init__(config, dataset, tokenizer)
        self.n_digit = tokenizer.n_digit
        self.codebook_size = tokenizer.codebook_size
        self.mask_token_id = tokenizer.vocab_size
        self.ignore_index = tokenizer.ignored_label
        self.temperature = config["temperature"]
        self.do_norm_and_scale = config["do_norm_and_scale"]
        self.label_smoothing = config.get("label_smoothing", 0.0)

        self.item_id2tokens = self._map_item_tokens().to(self.config["device"])

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size + 1,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config["n_embd"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_inner=config["n_inner"],
            activation_function=config["activation_function"],
            resid_pdrop=config["resid_pdrop"],
            embd_pdrop=config["embd_pdrop"],
            attn_pdrop=config["attn_pdrop"],
            layer_norm_epsilon=config["layer_norm_epsilon"],
            initializer_range=config["initializer_range"],
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)

        self.denoiser = Denoiser(
            n_digit=self.n_digit,
            n_embd=config["n_embd"],
            vocab_size=tokenizer.vocab_size + 1,
            mask_token_id=self.mask_token_id,
            n_layers=config["diffusion_layers"],
            n_heads=config["diffusion_heads"],
            dropout=config["dropout"],
            do_norm_and_scale=self.do_norm_and_scale,
        )
        self.denoiser.init_target_embeddings(self.gpt2.wte.weight)

        self.mask_counts = self._resolve_mask_counts()
        self.diffusion_ocn_strategy = config.get("diffusion_ocn_strategy", "static")
        if self.diffusion_ocn_strategy not in {"static", "random"}:
            raise ValueError("diffusion_ocn_strategy must be 'static' or 'random'.")
        self.diffusion_final_logits = config.get("diffusion_final_logits", "last_masked")
        if self.diffusion_final_logits not in {"last_masked", "extra_refinement"}:
            raise ValueError("diffusion_final_logits must be 'last_masked' or 'extra_refinement'.")

        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config["chunk_size"]
        self.num_beams = config["num_beams"]
        self.n_edges = config["n_edges"]
        self.propagation_steps = config["propagation_steps"]

    def _map_item_tokens(self) -> torch.Tensor:
        item_id2tokens = torch.zeros((self.dataset.n_items, self.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def _parse_sequence_config(self, value) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"none", "null", "~"}:
                return None
            raw_parts = [part.strip() for part in value.split(",") if part.strip()]
            return [float(part) for part in raw_parts] or None
        if isinstance(value, Iterable):
            parsed = [float(part) for part in value]
            return parsed or None
        return [float(value)]

    def _resolve_mask_counts(self) -> list[int]:
        explicit_counts = self._parse_sequence_config(self.config.get("diffusion_mask_counts"))
        if explicit_counts is not None:
            raw_counts = [int(count) for count in explicit_counts]
        else:
            ratios = self._parse_sequence_config(self.config.get("diffusion_mask_ratios")) or [1.0, 0.75, 0.5, 0.25, 0.125]
            raw_counts = [int(math.ceil(self.n_digit * ratio)) for ratio in ratios]
            raw_counts.append(int(self.config.get("diffusion_min_masks", 1)))

        counts = sorted({max(1, min(self.n_digit, count)) for count in raw_counts}, reverse=True)
        if not counts or counts[0] != self.n_digit:
            counts.insert(0, self.n_digit)
        min_masks = int(self.config.get("diffusion_min_masks", 1))
        min_masks = max(1, min(self.n_digit, min_masks))
        if counts[-1] != min_masks:
            counts.append(min_masks)
            counts = sorted(set(counts), reverse=True)
        return counts

    def _encode_history(self, batch: dict):
        input_tokens = batch["history_sid"]
        tok_emb = self.gpt2.wte(input_tokens)
        input_embs = tok_emb.mean(dim=-2)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch["history_mask"].long(),
        )
        memory_context = outputs.last_hidden_state
        memory_padding_mask = ~batch["history_mask"]
        return outputs, memory_context, memory_padding_mask

    def _global_to_local_labels(self, target_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.n_digit, device=target_tokens.device) * self.codebook_size + 1
        valid = target_tokens != self.ignore_index
        labels = torch.full_like(target_tokens, self.ignore_index)
        active = mask & valid
        labels[active] = target_tokens[active] - offsets.unsqueeze(0).expand_as(target_tokens)[active]
        return labels

    def _ocn_order(
        self,
        target_tokens: torch.Tensor,
        memory_context: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = target_tokens.size(0)
        if self.diffusion_ocn_strategy == "random":
            return torch.argsort(torch.rand(batch_size, self.n_digit, device=target_tokens.device), dim=1)

        full_mask = torch.ones(batch_size, self.n_digit, dtype=torch.bool, device=target_tokens.device)
        safe_targets = target_tokens.masked_fill(target_tokens == self.ignore_index, 0)
        probe_tokens = safe_targets.masked_fill(full_mask, self.mask_token_id)

        was_training = self.denoiser.training
        self.denoiser.eval()
        with torch.no_grad():
            logits = self.forward_denoiser_only(
                {
                    "target_tokens": probe_tokens,
                    "memory_context": memory_context,
                    "memory_padding_mask": memory_padding_mask,
                }
            )["logits"]
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values
        if was_training:
            self.denoiser.train()

        confidence = confidence.masked_fill(target_tokens == self.ignore_index, -1e9)
        return torch.argsort(confidence, dim=1, descending=True)

    def _make_training_views(
        self,
        target_tokens: torch.Tensor,
        memory_context: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = target_tokens.size(0)
        safe_targets = target_tokens.masked_fill(target_tokens == self.ignore_index, 0)
        order = self._ocn_order(target_tokens, memory_context, memory_padding_mask)

        all_target_tokens = []
        all_labels = []
        for count in self.mask_counts:
            cur_mask = torch.ones(batch_size, self.n_digit, dtype=torch.bool, device=target_tokens.device)
            n_reveal = self.n_digit - count
            if n_reveal > 0:
                reveal_cols = order[:, :n_reveal]
                cur_mask.scatter_(1, reveal_cols, False)

            cur_tokens = safe_targets.clone()
            cur_tokens[cur_mask] = self.mask_token_id
            cur_labels = self._global_to_local_labels(target_tokens, cur_mask)

            all_target_tokens.append(cur_tokens)
            all_labels.append(cur_labels)

        target_tokens_multi = torch.cat(all_target_tokens, dim=0)
        labels_multi = torch.cat(all_labels, dim=0)
        memory_context_multi = memory_context.repeat(len(self.mask_counts), 1, 1)
        memory_mask_multi = memory_padding_mask.repeat(len(self.mask_counts), 1)
        return target_tokens_multi, labels_multi, memory_context_multi, memory_mask_multi

    def _macro_view_loss(self, logits: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        view_losses = []
        for view_index in range(len(self.mask_counts)):
            start = view_index * batch_size
            end = start + batch_size
            view_logits = logits[start:end].reshape(-1, self.codebook_size)
            view_labels = labels[start:end].reshape(-1)
            valid_count = (view_labels != self.ignore_index).sum()
            if valid_count.item() == 0:
                continue
            loss = F.cross_entropy(
                view_logits,
                view_labels,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
                reduction="sum",
            )
            view_losses.append(loss / valid_count)

        if not view_losses:
            return logits.sum() * 0.0
        return torch.stack(view_losses).mean()

    def forward(self, batch: dict, return_loss=True):
        outputs, memory_context, memory_padding_mask = self._encode_history(batch)

        if not return_loss:
            outputs.memory_context = memory_context
            outputs.memory_padding_mask = memory_padding_mask
            return outputs

        target_tokens = batch["decoder_labels"].clone()
        batch_size = target_tokens.size(0)
        target_tokens_multi, labels_multi, memory_context_multi, memory_mask_multi = self._make_training_views(
            target_tokens,
            memory_context,
            memory_padding_mask,
        )
        denoiser_outputs = self.forward_denoiser_only(
            {
                "target_tokens": target_tokens_multi,
                "memory_context": memory_context_multi,
                "memory_padding_mask": memory_mask_multi,
            }
        )
        outputs.loss = self._macro_view_loss(denoiser_outputs["logits"], labels_multi, batch_size)
        return outputs

    def forward_denoiser_only(self, batch: dict) -> dict:
        device = next(self.parameters()).device
        target_tokens = batch["target_tokens"].to(device)
        memory_context = batch["memory_context"].to(device)
        memory_padding_mask = batch["memory_padding_mask"].to(device)

        states = self.denoiser(target_tokens, memory_context, memory_padding_mask)
        if self.do_norm_and_scale:
            states = F.normalize(states, dim=-1, eps=1e-8)

        codebook_end = self.n_digit * self.codebook_size + 1
        token_embs = self.gpt2.wte.weight[1:codebook_end].view(self.n_digit, self.codebook_size, -1)
        if self.do_norm_and_scale:
            token_embs = F.normalize(token_embs, dim=-1, eps=1e-8)

        logits = []
        for digit in range(self.n_digit):
            logit = torch.matmul(states[:, digit, :], token_embs[digit].T)
            if self.do_norm_and_scale:
                logit = logit / self.temperature
            logits.append(logit)
        return {"hidden_states": states, "logits": torch.stack(logits, dim=1)}

    def _step_to_next_mask_count(
        self,
        current_targets: torch.Tensor,
        logits: torch.Tensor,
        next_count: int,
    ) -> torch.Tensor:
        is_masked = current_targets == self.mask_token_id
        offsets = torch.arange(self.n_digit, device=current_targets.device) * self.codebook_size + 1
        probs = F.softmax(logits, dim=-1)
        confidence, pred_ids = probs.max(dim=-1)
        global_pred_ids = pred_ids + offsets.unsqueeze(0)

        next_targets = torch.where(is_masked, global_pred_ids, current_targets)
        confidence = confidence.masked_fill(~is_masked, 1e9)
        if next_count > 0:
            keep_masked = torch.topk(confidence, k=next_count, dim=-1, largest=False).indices
            next_targets.scatter_(1, keep_masked, self.mask_token_id)
        return next_targets

    def _token_logits_from_denoising(self, memory_context: torch.Tensor, memory_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = memory_context.size(0)
        current_targets = torch.full(
            (batch_size, self.n_digit),
            self.mask_token_id,
            dtype=torch.long,
            device=memory_context.device,
        )

        final_logits = None
        for index, count in enumerate(self.mask_counts):
            denoiser_outputs = self.forward_denoiser_only(
                {
                    "target_tokens": current_targets,
                    "memory_context": memory_context,
                    "memory_padding_mask": memory_padding_mask,
                }
            )
            logits = denoiser_outputs["logits"]

            if index == len(self.mask_counts) - 1:
                final_logits = logits
                break

            current_targets = self._step_to_next_mask_count(
                current_targets,
                logits,
                next_count=self.mask_counts[index + 1],
            )

        if self.diffusion_final_logits == "extra_refinement":
            current_targets = self._step_to_next_mask_count(current_targets, final_logits, next_count=0)
            final_logits = self.forward_denoiser_only(
                {
                    "target_tokens": current_targets,
                    "memory_context": memory_context,
                    "memory_padding_mask": memory_padding_mask,
                }
            )["logits"]

        return F.log_softmax(final_logits, dim=-1).reshape(batch_size, -1)

    def build_ii_sim_mat(self):
        n_items = self.dataset.n_items
        codebook_end = self.n_digit * self.codebook_size + 1
        token_embs = self.gpt2.wte.weight[1:codebook_end].view(self.n_digit, self.codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1, eps=1e-8)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)

        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            tokens_i = self.item_id2tokens[i_start:i_end]
            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                tokens_j = self.item_id2tokens[j_start:j_end]
                sum_block = torch.zeros((i_end - i_start, j_end - j_start), device=self.gpt2.device)

                for digit in range(self.n_digit):
                    rows = tokens_i[:, digit] - digit * self.codebook_size - 1
                    cols = tokens_j[:, digit] - digit * self.codebook_size - 1
                    sum_block += token_sims_01[digit].index_select(0, rows).index_select(1, cols)

                item_item_sim[i_start:i_end, j_start:j_end] = sum_block / self.n_digit
        return item_item_sim

    build_adjacency_list = RPG.build_adjacency_list
    init_graph = RPG.init_graph
    graph_propagation = RPG.graph_propagation

    def generate(self, batch, n_return_sequences=1):
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                outputs = self.forward(batch, return_loss=False)
                token_logits = self._token_logits_from_denoising(
                    outputs.memory_context,
                    outputs.memory_padding_mask,
                )

                if self.generate_w_decoding_graph:
                    if not self.init_flag:
                        self.init_graph()
                        self.init_flag = True
                    return self.graph_propagation(token_logits, n_return_sequences)

                item_logits = torch.gather(
                    input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
                    dim=-1,
                    index=(self.item_id2tokens[1:, :] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1),
                ).mean(dim=-1)
                preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
                return preds.unsqueeze(-1)
        finally:
            if was_training:
                self.train()
