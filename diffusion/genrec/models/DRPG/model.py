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


class Denoiser(nn.Module):
    def __init__(self, n_digit, n_embd, vocab_size, mask_token_id, n_layers, n_heads, dropout, do_norm_and_scale):
        super().__init__()
        self.n_digit = n_digit
        self.n_embd = n_embd
        self.mask_token_id = mask_token_id

        self.target_embeddings = nn.Embedding(vocab_size, n_embd)
        # Separate mask embedding per position so no positional embeddings, like in diffGRM
        self.mask_embeddings = nn.Embedding(n_digit, n_embd)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )

        final_norm = nn.LayerNorm(n_embd)
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
            norm=final_norm,
        )

        self.do_norm_and_scale = do_norm_and_scale


    def init_target_embeddings(self, gpt2_wte_weights):
        # Share embedding table weights with encoder
        self.target_embeddings.weight = gpt2_wte_weights
        with torch.no_grad():
            self.mask_embeddings.weight.normal_(0, 0.02)

    def forward(self, target_tokens, user_history, memory_padding_mask):
        """
        Args:
            target_tokens (batch_size, n_digit): (Partially) masked target token IDs
            user_history (batch_size, seq_len, n_embd): Full sequence context history
            memory_padding_mask (batch_size, seq_len): Boolean mask where True blocks attention
        """
        B = target_tokens.size(0)

        is_masked = (target_tokens == self.mask_token_id) # (B, n_digit)

        # Get token embeddings for unmasked tokens
        safe_tokens = target_tokens.masked_fill(is_masked, 0)
        token_embs = self.target_embeddings(safe_tokens) # (B, n_digit, n_embd)
        if self.do_norm_and_scale:
            token_embs = F.normalize(token_embs, dim=-1, eps=1e-8)

        # Get mask embeddings for masked tokens
        pos_ids = torch.arange(self.n_digit, device=target_tokens.device)
        mask_embs = self.mask_embeddings(pos_ids).unsqueeze(0).expand(B, -1, -1) # (B, n_digit, n_embd)
        if self.do_norm_and_scale:
            mask_embs = F.normalize(mask_embs, dim=-1, eps=1e-8)

        # Set token/ mask embeddings
        tgt = torch.where(is_masked.unsqueeze(-1), mask_embs, token_embs)

        output_states = self.transformer(
            tgt=tgt,
            memory=user_history,
            memory_key_padding_mask=memory_padding_mask
        )

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

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size + 1,  # +1 for the MASK token
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
        self.do_norm_and_scale = config['do_norm_and_scale']

        self.denoiser = Denoiser(
            n_digit=self.n_digit,
            n_embd=config['n_embd'],
            vocab_size=tokenizer.vocab_size,
            mask_token_id=self.mask_token_id,
            n_layers=config['diffusion_layers'],
            n_heads=config['diffusion_heads'],
            dropout=config['dropout'],
            do_norm_and_scale=self.do_norm_and_scale,
        )
        # Share embedding weights
        self.denoiser.init_target_embeddings(self.gpt2.wte.weight)

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label, label_smoothing=config['label_smoothing'])  # Label smoothing like in eq. 3 from https://arxiv.org/pdf/2510.21805

        # Graph-constrained decoding
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        self.oracle_generate = self.config['oracle_generate']
        self.n_oracle = self.config["n_oracle"]
        self.log_pred_acc = self.config["log_pred_acc"]
        print(f"Using oracle generation: {self.oracle_generate} with {self.n_oracle} digits.")
        print(f"Logging token prediction accuracy: {self.log_pred_acc}")

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
        input_tokens = batch['history_sid']
        tok_emb = self.gpt2.wte(input_tokens)

        input_embs = tok_emb.mean(dim=-2)

        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['history_mask'].long(),  # 1=valid, 0=pad for GPT2 encoder
        )  # outputs.last_hidden_state: (B, seq_len, n_embed)

        if return_loss:
            memory_context = outputs.last_hidden_state  # (B, seq_len, n_embd)
            memory_padding_mask = ~batch['history_mask']  # (B, seq_len)

            target_tokens = batch['decoder_labels'].clone()
            B = target_tokens.size(0)

            # On-Policy Confidence Estimation (OCN)
            _was_training = self.denoiser.training
            self.denoiser.eval()

            offsets = torch.arange(self.n_digit, device=target_tokens.device) * self.config['codebook_size'] + 1

            def score_with_mask(mask_state: torch.Tensor) -> torch.Tensor:
                """Scores confidence given an arbitrary mixture of masks and true values."""
                temp_tokens = target_tokens.clone()
                temp_tokens[mask_state] = self.denoiser.mask_token_id

                denoiser_outputs = self.forward_denoiser_only({
                    'target_tokens': temp_tokens,
                    'memory_context': memory_context,
                    'memory_padding_mask': memory_padding_mask
                })

                # logits = torch.clamp(denoiser_outputs['logits'], min=-50.0, max=50.0)
                logits = denoiser_outputs['logits']
                probs = F.softmax(logits, dim=-1)

                conf = probs.max(dim=-1).values

                return conf

            all_target_tokens = []
            all_labels = []

            n_views = self.config['n_views']
            mask_steps = torch.linspace(1, self.n_digit, steps=n_views).long().tolist()
            # Sort descending to build a trajectory from heavy corruption to light corruption
            mask_steps_descending = sorted(mask_steps, reverse=True)

            # Start with a fully masked tracking state (True = Masked)
            cur_mask = torch.ones((B, self.n_digit), dtype=torch.bool, device=target_tokens.device)

            with torch.no_grad():
                for i, num_to_mask in enumerate(mask_steps_descending):
                    # 2. Build and store training inputs/labels for this view state
                    cur_tokens = target_tokens.clone()
                    cur_tokens[cur_mask] = self.denoiser.mask_token_id

                    cur_labels = target_tokens.clone()
                    cur_labels = cur_labels - offsets.unsqueeze(0)
                    cur_labels[~cur_mask] = self.loss_fct.ignore_index

                    all_target_tokens.append(cur_tokens)
                    all_labels.append(cur_labels)

                    # 3. Update the mask dynamically for the next lighter view step
                    if i < len(mask_steps_descending) - 1:
                        next_num_to_mask = mask_steps_descending[i + 1]
                        num_to_reveal = num_to_mask - next_num_to_mask

                        # Guard against potential 0 step adjustments from integer rounding
                        if num_to_reveal > 0:
                            # Contextual evaluation on the current arbitrary mask layout
                            confidence = score_with_mask(cur_mask)

                            # Ignore already unmasked slots by penalizing their confidence
                            confidence[~cur_mask] = -1e9

                            # Reveal the slots the model is currently most certain about
                            _, cols_to_reveal = torch.topk(confidence, k=num_to_reveal, dim=-1, largest=True)
                            cur_mask.scatter_(1, cols_to_reveal, False)

            if _was_training:
                self.denoiser.train()

            target_tokens_multi = torch.cat(all_target_tokens, dim=0)  # (B * n_views, n_digit)
            shifted_labels_multi = torch.cat(all_labels, dim=0)  # (B * n_views, n_digit)

            memory_context_multi = memory_context.repeat(n_views, 1, 1)  # (B * n_views, seq_len, d)
            memory_mask_multi = memory_padding_mask.repeat(n_views, 1)  # (B * n_views, seq_len)

            denoiser_outputs = self.forward_denoiser_only({
                'target_tokens': target_tokens_multi,
                'memory_context': memory_context_multi,
                'memory_padding_mask': memory_mask_multi
            })

            logits = denoiser_outputs['logits']  # (B_multi, n_digit, codebook_size)

            # micro-averaging loss like diffGRM instead of macro-avg loss in RPG, slightly more stable because not every digit is masked an equal number of time
            logits_flat = logits.view(-1, self.config['codebook_size'])
            labels_flat = shifted_labels_multi.view(-1)

            outputs.loss = self.loss_fct(logits_flat, labels_flat)
        # If not returning loss/ training, only return last hidden state in sequence
        # in order to predict with all context. Decoding is done in generate()
        else:
            outputs.memory_context = outputs.last_hidden_state  # (B, seq_len, n_embd)
            outputs.memory_padding_mask = ~batch['history_mask']  # (B, seq_len)

        return outputs

    def forward_denoiser_only(self, batch: dict) -> dict:
        """
        Runs only the denoiser (decoder) part of the model.

        Args:
            batch: Dictionary containing:
                - target_tokens: (B, n_digit)
                - memory_context: (B, seq_len, n_embd)
                - memory_padding_mask: (B, seq_len)

        Returns:
            A dictionary containing 'hidden_states' and 'logits'.
        """
        device = next(self.parameters()).device

        target_tokens = batch['target_tokens'].to(device)  # decoder_inputs in DiffGRM
        memory_context = batch['memory_context'].to(device)  # encoder_hidden in DiffGRM
        memory_padding_mask = batch['memory_padding_mask'].to(device)  # Different from DiffGRM's mask_positions. Masks are already in target_tokens and read in Denoiser.forward()

        # Pass through the denoiser
        states = self.denoiser(target_tokens, memory_context, memory_padding_mask)
        if self.do_norm_and_scale:
            states = F.normalize(states, dim=-1, eps=1e-8)

        # Extract token embeddings for logit computation
        codebook_range = self.n_digit * self.config['codebook_size'] + 1
        token_emb = self.gpt2.wte.weight[1:codebook_range]
        if self.do_norm_and_scale:
            token_emb = F.normalize(token_emb, dim=-1, eps=1e-8)
        token_embs = torch.chunk(token_emb, self.n_digit, dim=0)

        # Compute logits per digit
        logits = []
        for i in range(self.n_digit):
            if self.do_norm_and_scale:
                logit = torch.matmul(states[:, i, :], token_embs[i].T) / self.temperature
            else:
                logit = torch.matmul(states[:, i, :], token_embs[i].T)
            logits.append(logit)

        logits_stack = torch.stack(logits, dim=1)  # (B, n_digit, codebook_size)

        return {
            "hidden_states": states,
            "logits": logits_stack
        }

    def build_ii_sim_mat(self):
        # Assuming n_digit=32, codebook_size=256
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)
        # token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
        codebook_range = self.n_digit * self.config['codebook_size'] + 1
        token_embs = self.gpt2.wte.weight[1:codebook_range].view(n_digit, codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        token_embs = F.normalize(token_embs, dim=-1, eps=1e-8)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # 4) Prepare an output similarity matrix
        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # shape: (chunk_i_size, 32)
            tokens_i = self.item_id2tokens[i_start:i_end].to(self.gpt2.device)  # sub-block for items i

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # shape: (chunk_j_size, 32)
                tokens_j = self.item_id2tokens[j_start:j_end].to(self.gpt2.device)  # sub-block for items j

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
        was_training = self.training

        if was_training:
            self._total_correct_guesses = 0
            self._total_free_tokens = 0

        self.eval()
        try:
            with torch.no_grad():
                outputs = self.forward(batch, return_loss=False)
                memory_context = outputs.memory_context
                memory_padding_mask = outputs.memory_padding_mask
                B = memory_context.size(0)
                device = memory_context.device

                denoise_steps = self.config['denoise_inference_steps']

                # Target starts fully masked
                current_targets = torch.full(
                    (B, self.n_digit),
                    self.denoiser.mask_token_id,
                    device=device
                )

                n_oracle = 0
                n_free = self.n_digit
                oracle_active = self.oracle_generate  # For diffusion multistep explanation tests
                log_pred_acc = self.log_pred_acc  # For diffusion multistep explanation tests

                if (oracle_active or log_pred_acc) and 'labels' in batch:
                    target_item_ids = batch['labels'].view(-1)
                    true_target_tokens = self.item_id2tokens[target_item_ids]

                    if oracle_active:
                        n_oracle = self.n_oracle
                        n_free = self.n_digit - n_oracle

                        # Reveal the first n_oracle tokens
                        current_targets[:, :n_oracle] = true_target_tokens[:, :n_oracle]

                offsets = torch.arange(self.n_digit, device=device) * self.config['codebook_size'] + 1

                final_logits = None
                steps = min(denoise_steps, self.n_digit)

                for step in range(1, steps+1):
                    is_masked = (current_targets == self.denoiser.mask_token_id)

                    denoiser_outputs = self.forward_denoiser_only({
                        'target_tokens': current_targets,
                        'memory_context': memory_context,
                        'memory_padding_mask': memory_padding_mask
                    })

                    logits = denoiser_outputs['logits']

                    if step == steps:
                        final_logits = logits.clone()  # (B, n_digit, codebook_size)

                    # Get highest confidence digits
                    probs = torch.softmax(logits, dim=-1)
                    max_probs, pred_ids = probs.max(dim=-1)
                    confidence = max_probs.clone()
                    confidence[~is_masked] = 1e9

                    global_pred_ids = pred_ids + offsets.unsqueeze(0)

                    if step == steps:
                        # If it's the final step, unmask everything that is left. 0 tokens remain masked.
                        num_to_mask = 0
                    else:
                        if oracle_active:
                            # Unmask an even chunk of the free tokens per step
                            unmask_per_step = n_free // steps
                            num_to_mask = n_free - (step * unmask_per_step)
                        else:
                            unmask_per_step = self.n_digit // steps
                            num_to_mask = self.n_digit - (step * unmask_per_step)

                    next_targets = torch.where(is_masked, global_pred_ids, current_targets)
                    if num_to_mask > 0:
                        # Keep lowest confidence digits masked
                        mask_idx = torch.topk(confidence, k=num_to_mask, dim=-1, largest=False).indices
                        next_targets.scatter_(1, mask_idx, self.denoiser.mask_token_id)

                    # Re-enforce true targets if oracle is active
                    if oracle_active and 'labels' in batch:
                        next_targets[:, :n_oracle] = true_target_tokens[:, :n_oracle]

                    current_targets = next_targets

                # Keep track of token prediction accuracy
                if log_pred_acc and 'labels' in batch:
                    if not hasattr(self, '_total_correct_guesses'):
                        self._total_correct_guesses = 0
                        self._total_free_tokens = 0

                    # Create a mask so we only calculate accuracy on unmasked tokens
                    free_token_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                    free_token_mask[:, :n_oracle] = False

                    correct_guesses = (current_targets == true_target_tokens) & free_token_mask

                    if free_token_mask.sum() > 0:
                        self._total_correct_guesses += correct_guesses.sum().item()
                        self._total_free_tokens += free_token_mask.sum().item()

                token_logits = F.log_softmax(final_logits, dim=-1).view(B, -1)

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
                        input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
                        dim=-1,
                        index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
                    ).mean(dim=-1)

                    preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
                    return preds.unsqueeze(-1)
        finally:
            if was_training:
                self.train()
