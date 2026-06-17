# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])
        # For duplicates metric
        self.total_seqs = 0
        self.total_unique = 0
        self.batch_duplicate_ratios = []
        self.batch_dup10_ratios = []

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def _dup_ratio_per_user(self, preds_cpu, k=10):
        """
        Calculates the internal duplicate ratio within a single user's top-k recommendations.
        """
        B = preds_cpu.shape[0]
        actual_k = min(k, preds_cpu.shape[1])  # Fallback if maxk < k
        dup_ratios = []

        for b in range(B):
            seqs = preds_cpu[b, :actual_k]
            # Convert to tuples to find unique sequences via a set
            k_seqs = [tuple(s.tolist()) for s in seqs]
            unique_cnt = len(set(k_seqs))
            dup_ratios.append(1.0 - (unique_cnt / actual_k))

        return torch.tensor(dup_ratios, dtype=torch.float32)

    def calculate_metrics(self, preds, labels):
        if isinstance(preds, tuple):
            preds_tensor, n_visited_items = preds
        else:
            preds_tensor = preds
            n_visited_items = torch.FloatTensor([len(self.tokenizer.item2tokens)] * preds_tensor.shape[0])

        results = {}
        pos_index = self.calculate_pos_index(preds_tensor, labels)

        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        results['n_visited_items'] = n_visited_items

        preds_cpu = preds_tensor.detach().cpu()
        B, maxk = preds_cpu.shape[0], preds_cpu.shape[1]

        # Global / Batch Duplicate Ratio
        flat_preds = preds_cpu.reshape(B * maxk, -1)
        unique_seqs = len(set(tuple(s.tolist()) for s in flat_preds))
        total_seqs_in_batch = B * maxk

        current_duplicate_ratio = 1.0 - (unique_seqs / total_seqs_in_batch)
        results['duplicate_ratio'] = torch.tensor([current_duplicate_ratio], dtype=torch.float32)

        # User-level Top-10 Duplicate Ratio
        dup10 = self._dup_ratio_per_user(preds_cpu, k=10)
        results['dup@10'] = dup10

        self.total_seqs += total_seqs_in_batch
        self.total_unique += unique_seqs
        self.batch_duplicate_ratios.append(current_duplicate_ratio)
        self.batch_dup10_ratios.append(dup10.mean().item())

        return results

    def print_final_stats(self):
        """Prints overall statistics collected across all batches."""
        if self.total_seqs > 0:
            overall_duplicate_ratio = 1.0 - (self.total_unique / self.total_seqs)

            if self.batch_duplicate_ratios:
                avg_duplicate_ratio = sum(self.batch_duplicate_ratios) / len(self.batch_duplicate_ratios)
                print(f"[EVAL_STATS] Avg Batch Global Dup Ratio: {avg_duplicate_ratio:.3f}")

                if self.batch_dup10_ratios:
                    avg_dup10 = sum(self.batch_dup10_ratios) / len(self.batch_dup10_ratios)
                    print(f"[EVAL_STATS] Avg User Internal Dup@10: {avg_dup10:.3f}")
            else:
                print(f"[EVAL_STATS] Overall Global Dup Ratio: {overall_duplicate_ratio:.3f}")
