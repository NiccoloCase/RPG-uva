# Adapted from
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


# Technically not denoising but demasking, because discrete diffusion
class Denoiser(nn.Module):
    def __init__(self, n_digit, n_embd, vocab_size, n_layers, n_heads, dropout):
        super().__init__()
        self.n_digit = n_digit
        self.n_embd = n_embd
        self.mask_token_id = vocab_size  # Last index is [MASK] token

        self.target_embeddings = nn.Embedding(vocab_size, n_embd)
        # Seperate mask embedding per position so no positional embeddings, like in diffGRM
        self.mask_embeddings = nn.Embedding(n_digit, n_embd)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.final_proj = nn.Linear(n_embd, n_embd)

    def init_target_embeddings(self, gpt2_wte_weights):
        # Share embedding table weights with encoder
        self.target_embeddings.weight = gpt2_wte_weights
        with torch.no_grad():
            self.mask_embeddings.weight.normal_(0, 0.02)

    def forward(self, target_tokens, context):
        """
        Args:
            target_tokens (batch_size, n_digit): (Partially) masked target token IDs
            context (batch_size, n_embd): User history context vector
        """
        B = target_tokens.size(0)

        is_masked = (target_tokens == self.mask_token_id) # (B, n_digit)

        # Temporarily replace the mask_token_id with 0 to avoid index out-of-bounds in target_embeddings.
        safe_tokens = target_tokens.masked_fill(is_masked, 0)
        token_embs = self.target_embeddings(safe_tokens) # (B, n_digit, n_embd)

        pos_ids = torch.arange(self.n_digit, device=target_tokens.device)
        mask_embs = self.mask_embeddings(pos_ids).unsqueeze(0).expand(B, -1, -1) # (B, n_digit, n_embd)

        # Mask where needed
        tgt = torch.where(is_masked.unsqueeze(-1), mask_embs, token_embs)

        # TODO: check if this is still needed
        if context.dim() == 2:
            memory = context.unsqueeze(1)  # (B, 1, n_embd)
        else:
            memory = context               # (B, seq_len, n_embd)

        output_states = self.transformer(tgt=tgt, memory=memory)  # (B, n_digit, n_embd)
        output_states = self.final_proj(output_states)

        return output_states


class DRPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(DRPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        # TODO: Replace self.gpt with just an nn.Embedding() (with same hparams as self.gpt has) and a custom encoder
        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)

        self.n_digit = self.tokenizer.n_digit
        self.mask_token_id = tokenizer.vocab_size
        self.num_timesteps = config['num_timesteps']

        self.denoiser = Denoiser(
            n_digit=self.n_digit,
            n_embd=config['n_embd'],
            vocab_size=tokenizer.vocab_size,
            n_layers=config['diffusion_layers'],
            n_heads=config['diffusion_heads'],
            dropout=config['dropout']
        )
        # Share embedding weights
        self.denoiser.init_target_embeddings(self.gpt2.wte.weight)

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label, label_smoothing=0.1)  # Label smoothing like in eq. 3 from https://arxiv.org/pdf/2510.21805

        # Graph-constrained decoding
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']


    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_tokens = self.item_id2tokens[batch['input_ids']]  # (B, seq_len, n_codebook)
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)  # (B, seq_len, n_embd)
        B = input_embs.size(0)

        # Run encoder
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )  # outputs.last_hidden_state: (B, seq_len_ n_embed)

        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            target_labels = self.item_id2tokens[batch['labels']]  # (B, seq_len, n_digit)
            label_mask = batch['labels'].view(-1) != -100  # (B*seq_len)

            selected_context = outputs.last_hidden_state.view(-1, self.config['n_embd'])[label_mask]  # (B*seq_len, n_embed)
            selected_target_labels = target_labels.view(-1, self.n_digit)[label_mask]  # (B*seq_len, n_digit)

            B_eff = selected_target_labels.size(0)  # (B*seq_len)

            target_tokens = selected_target_labels.clone()  # (B*seq_len, n_digit)

            # On-Policy Confidence Estimation (OCN). See section 3.3 in https://arxiv.org/pdf/2510.21805
            # "we compute the difficulty order once per example with a single fully masked pass, reuse the encoder output across the 𝑅 views"
            with torch.no_grad():
                # "run the MD-Decoder once on a fully masked 𝑛-digit input"
                dummy_masked = torch.full_like(target_tokens, self.denoiser.mask_token_id)

                baseline_states = self.denoiser(dummy_masked, selected_context)
                baseline_states = F.normalize(baseline_states, dim=-1, eps=1e-8)
                baseline_states_chunked = torch.chunk(baseline_states, self.n_digit, dim=1)

                token_emb_eval = self.gpt2.wte.weight[1:-1].detach()
                token_emb_eval = F.normalize(token_emb_eval, dim=-1, eps=1e-8)
                token_embs_eval_chunked = torch.chunk(token_emb_eval, self.n_digit, dim=0)

                confidence = torch.zeros((B_eff, self.n_digit), device=target_tokens.device)

                for k in range(self.n_digit):
                    offsets_k = k * self.config['codebook_size'] + 1
                    local_labels_k = selected_target_labels[:, k] - offsets_k

                    logits_i = torch.matmul(baseline_states_chunked[k].squeeze(1), token_embs_eval_chunked[k].T) / self.temperature
                    logits_i = torch.clamp(logits_i, min=-50.0, max=50.0)
                    probs_i = F.softmax(logits_i, dim=-1)

                    # Extract the probability of the correct token
                    # confidence[:, k] = probs_i.max(dim=-1).values
                    confidence[:, k] = probs_i.gather(dim=-1, index=local_labels_k.unsqueeze(-1)).squeeze(-1)

            uncertainty = 1.0 - confidence
            mask_weights = uncertainty + 1e-5  # Tiny epsilon so high-confidence tokens have a non-zero chance of being masked.

            # "OCN constructs a small nested set of views per sample"
            n_views = self.config['n_views']
            B_multi = B_eff * n_views

            # R / n_view different views
            mask_weights_multi = mask_weights.repeat_interleave(n_views, dim=0)
            selected_context_multi = selected_context.repeat_interleave(n_views, dim=0)
            target_tokens_multi = target_tokens.repeat_interleave(n_views, dim=0)
            selected_target_labels_multi = selected_target_labels.repeat_interleave(n_views, dim=0)

            num_digits_to_mask = torch.randint(1, self.n_digit + 1, (B_multi,), device=target_tokens.device)

            # Generate Gumbel noise to sample proportional to weights
            U = torch.rand_like(mask_weights_multi).clamp(1e-5, 1.0 - 1e-5)
            gumbel_scores = torch.log(mask_weights_multi) - torch.log(-torch.log(U) + 1e-5)

            # "... ordered from light to heavy corruption"
            ranks = gumbel_scores.argsort(dim=-1, descending=True).argsort(dim=-1)

            to_mask = ranks < num_digits_to_mask.unsqueeze(-1)  # (B_multi, n_digit)
            target_tokens_multi[to_mask] = self.denoiser.mask_token_id

            # Do not calculate loss for unmasked tokens (M^{r} in eq. 3 in https://arxiv.org/pdf/2510.21805)
            offsets = torch.arange(self.n_digit, device=target_tokens.device) * self.config['codebook_size'] + 1
            shifted_labels_multi = selected_target_labels_multi - offsets.unsqueeze(0)
            shifted_labels_multi[~to_mask] = self.loss_fct.ignore_index

            # Demask
            final_states = self.denoiser(target_tokens_multi, selected_context_multi)
            final_states = F.normalize(final_states, dim=-1, eps=1e-8)
            final_states = torch.chunk(final_states, self.n_digit, dim=1)

            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1, eps=1e-8)
            token_embs = torch.chunk(token_emb, self.n_digit, dim=0)

            token_logits = [torch.matmul(final_states[k].squeeze(dim=1), token_embs[k].T) / self.temperature for k in range(self.n_digit)]
            token_logits = [torch.clamp(logit, min=-50.0, max=50.0) for logit in token_logits]  # prevent nan

            # The standard loss mean over B_multi naturally acts as the 1/|M| average over views described in eq. 3 in https://arxiv.org/pdf/2510.21805
            losses = [self.loss_fct(token_logits[k], shifted_labels_multi[:, k]) for k in range(self.n_digit)]

            outputs.loss = torch.mean(torch.stack(losses))
        # If not returning loss/ training, only return last hidden state in sequence
        # in order to predict with all context. Decoding is done in generate()
        else:
            seq_lens = batch['seq_lens'] - 1
            gather_index = seq_lens.view(-1, 1, 1).expand(-1, 1, self.config['n_embd'])
            context = outputs.last_hidden_state.gather(dim=1, index=gather_index).squeeze(1)
            outputs.final_states = context  # (B, n_embd)

        return outputs

    def build_ii_sim_mat(self):
        # Assuming n_digit=32, codebook_size=256
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # 4) Prepare an output similarity matrix
        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # shape: (chunk_i_size, 32)
            tokens_i = self.item_id2tokens[i_start:i_end]  # sub-block for items i

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # shape: (chunk_j_size, 32)
                tokens_j = self.item_id2tokens[j_start:j_end]  # sub-block for items j

                # We want to compute a sub-block of shape: (chunk_i_size, chunk_j_size).
                # For each digit k in [0..31], we look up token_sims_01[k, tokens_i[i, k], tokens_j[j, k]].

                # We'll accumulate the similarity for each of the 32 digits
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)

                # We'll do a small loop over k=0..31 (which is constant = 32).
                # Each token_sims_01[k] is (256, 256). We gather from it using:
                #   row indices = tokens_i[:, k]
                #   col indices = tokens_j[:, k]
                #
                # The typical approach is:
                #   sub = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                # Then sum them up across k.
                for k in range(n_digit):
                    # row_inds shape: (block_size_i,)
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    # col_inds shape: (block_size_j,)
                    col_inds = tokens_j[:, k] - k * codebook_size - 1

                    # token_sims_01[k] -> shape (256, 256)
                    # row-gather => shape (block_size_i, 256)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    # col-gather across dim=1 => shape (block_size_i, block_size_j)
                    temp = temp.index_select(1, col_inds)

                    # Accumulate
                    sum_block += temp

                # Now take the average across the 32 digits
                avg_block = sum_block / n_digit

                # Write back into the final item_item_sim
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        batch_size = token_logits.shape[0]

        # Initialize visited nodes tracking
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Randomly sample num_beams distinct node IDs in [1..n_nodes]
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Add initial nodes to visited set
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        for sid in range(self.propagation_steps):
            # Find neighbors of these top num_beams nodes
            #      adjacency_list is 0-based internally => need node_id-1
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])

                # Add neighbors to visited set
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)

                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        # Convert visited counts to tensor
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        context = outputs.final_states
        B = context.size(0)
        device = context.device

        denoise_steps = self.config['denoise_inference_steps']

        # Target starts fully masked
        current_targets = torch.full(
            (B, self.n_digit),
            self.denoiser.mask_token_id,
            device=device
        )

        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_digit, dim=0)
        for step in range(1, denoise_steps + 1):
            states = self.denoiser(current_targets, context)  # (B, n_digit, n_embd)
            states = F.normalize(states, dim=-1)

            logits = [torch.matmul(states[:,i,:], token_embs[i].T) / self.temperature for i in range(self.n_digit)]

            # If final step, prepare logit matrix for Graph Search
            if step == denoise_steps:
                logits = [F.log_softmax(logit, dim=-1) for logit in logits]
                token_logits = torch.cat(logits, dim=-1)  # (B, n_digit * codebook_size)
                break

            # If intermediate step, lock in the most confident predictions
            logits_stack = torch.stack(logits, dim=1)  # (B, n_digit, codebook_size)
            probs = torch.softmax(logits_stack, dim=-1)
            max_probs, pred_ids = probs.max(dim=-1)  # max confidence, ID (0-codebook_size-1)

            # Local pred_id (0-codebook_size-1) to global vocab token ID
            offsets = torch.arange(self.n_digit, device=device) * self.config['codebook_size'] + 1
            global_pred_ids = pred_ids + offsets.unsqueeze(0)

            is_masked = (current_targets == self.denoiser.mask_token_id)

            confidence = max_probs.clone()
            confidence[~is_masked] = 1e9

            # Unmask with cosine strategy similar
            progress = step / denoise_steps
            ratio_to_mask = math.cos(progress * math.pi / 2.0)
            num_to_mask = max(0, int(self.n_digit * ratio_to_mask))

            if num_to_mask > 0:
                mask_idx = torch.topk(confidence, k=num_to_mask, dim=-1, largest=False).indices

                next_targets = current_targets.clone()
                next_targets[is_masked] = global_pred_ids[is_masked]
                next_targets.scatter_(1, mask_idx, self.denoiser.mask_token_id)
                current_targets = next_targets
            else:
                current_targets[is_masked] = global_pred_ids[is_masked]

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            outputs = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
            return outputs
        else:
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),              # (batch_size, n_items, n_digit)
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)  # (batch_size, n_items, code_dim)
            ).mean(dim=-1)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            return preds.unsqueeze(-1)
