from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .utils import ndcg_k, recall_at_k


class SASRecTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        )
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.weight_decay,
        )
        self.criterion = nn.BCELoss()
        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))

    def train(self, epoch: int) -> dict[str, float]:
        return self._iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch: int, full_sort: bool = True):
        return self._iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch: int, full_sort: bool = True):
        return self._iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def save(self, file_name: str) -> None:
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name: str) -> None:
        state_dict = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def cross_entropy(
        self,
        seq_out: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float()
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        return loss

    def full_softmax_cross_entropy(
        self,
        seq_out: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax cross-entropy over the whole catalogue at every target position.

        The objective of the "modern" SASRec recipe: each position's hidden
        state is scored against every item embedding and the next item is the
        softmax target. Padding (id 0) and the unused mask id are excluded from
        the softmax. Memory grows with positions x catalogue size, so this is
        meant for catalogues up to tens of thousands of items.
        """
        target_mask = pos_ids.view(-1) > 0
        hidden = seq_out.reshape(-1, seq_out.size(-1))[target_mask]
        logits = torch.matmul(hidden, self.model.item_embeddings.weight.transpose(0, 1))
        logits[:, 0] = float("-inf")
        if 0 <= self.args.mask_id < logits.shape[1]:
            logits[:, self.args.mask_id] = float("-inf")
        return nn.functional.cross_entropy(logits, pos_ids.view(-1)[target_mask])

    def sampled_softmax_cross_entropy(
        self,
        seq_out: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sampled estimate of the full-catalogue softmax cross-entropy.

        For catalogues too large for ``full_softmax_cross_entropy``. Each batch
        draws ``n_negatives`` items uniformly from the real items (ids
        1..max_item), shared by all target positions. Every position is scored
        against its target and these negatives; a negative equal to the target
        is masked out. Negative logits get the logQ correction
        ``log(n_items / n_negatives)``, so the sampled denominator is an
        unbiased estimate of the full softmax denominator.
        """
        target_mask = pos_ids.view(-1) > 0
        hidden = seq_out.reshape(-1, seq_out.size(-1))[target_mask]
        targets = pos_ids.view(-1)[target_mask]
        n_real_items = self.args.item_size - 2
        n_negatives = self.args.n_negatives
        negatives = torch.randint(1, n_real_items + 1, (n_negatives,), device=hidden.device)

        embeddings = self.model.item_embeddings.weight
        positive_logits = (hidden * embeddings[targets]).sum(-1, keepdim=True)
        negative_logits = hidden @ embeddings[negatives].T
        negative_logits = negative_logits + math.log(n_real_items / n_negatives)
        negative_logits = negative_logits.masked_fill(negatives.unsqueeze(0) == targets.unsqueeze(1), float("-inf"))

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        zeros = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return nn.functional.cross_entropy(logits, zeros)

    def predict_full(self, seq_out: torch.Tensor) -> torch.Tensor:
        test_item_emb = self.model.item_embeddings.weight
        return torch.matmul(seq_out, test_item_emb.transpose(0, 1))

    def _log(self, payload) -> None:
        print(payload)
        with open(self.args.log_file, "a", encoding="utf-8") as handle:
            handle.write(f"{payload}\n")

    def _format_eval_metrics(self, epoch: int, metrics: OrderedDict[str, float]) -> OrderedDict[str, str]:
        formatted = OrderedDict()
        formatted["Epoch"] = epoch
        for key, value in metrics.items():
            metric_name, k = key.split("@")
            label = f"{'RECALL' if metric_name == 'recall' else 'NDCG'}@{k}"
            formatted[label] = f"{value:.4f}"
        return formatted

    def _full_sort_metrics(
        self,
        epoch: int,
        answers: np.ndarray,
        pred_list: np.ndarray,
    ) -> tuple[OrderedDict[str, float], str]:
        metrics = OrderedDict()
        for k in self.args.topk:
            metrics[f"recall@{k}"] = recall_at_k(answers, pred_list, k)
            metrics[f"ndcg@{k}"] = ndcg_k(answers, pred_list, k)
        formatted = self._format_eval_metrics(epoch, metrics)
        self._log(formatted)
        return metrics, str(formatted)

    def _iteration(self, epoch: int, dataloader, full_sort: bool = True, train: bool = True):
        mode = "train" if train else "eval"
        data_iter = tqdm(
            enumerate(dataloader),
            desc=f"SASRec {mode} epoch {epoch}",
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if train:
            self.model.train()
            total_loss = 0.0
            current_loss = 0.0
            for _, batch in data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                sequence_output = self.model(input_ids)
                if self.args.loss_type == "ce":
                    loss = self.full_softmax_cross_entropy(sequence_output, target_pos)
                elif self.args.loss_type == "sampled_ce":
                    loss = self.sampled_softmax_cross_entropy(sequence_output, target_pos)
                else:
                    loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                current_loss = loss.item()

            payload = {
                "epoch": epoch,
                "rec_avg_loss": f"{(total_loss / max(len(dataloader), 1)):.4f}",
                "rec_cur_loss": f"{current_loss:.4f}",
            }
            if (epoch + 1) % self.args.log_freq == 0:
                self._log(payload)
            return {
                "rec_avg_loss": total_loss / max(len(dataloader), 1),
                "rec_cur_loss": current_loss,
            }

        if not full_sort:
            raise NotImplementedError("Only full-sort SASRec evaluation is supported in RPG-uva.")

        self.model.eval()
        pred_list = None
        answer_list = None
        topk_max = max(self.args.topk)
        with torch.no_grad():
            for _, batch in data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, _, answers = batch
                sequence_output = self.model(input_ids)
                recommend_output = sequence_output[:, -1, :]
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[:, 0] = -np.inf
                if 0 <= self.args.mask_id < rating_pred.shape[1]:
                    rating_pred[:, self.args.mask_id] = -np.inf
                if getattr(self.args, "mask_history", True):
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = -np.inf

                ind = np.argpartition(rating_pred, -topk_max)[:, -topk_max:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if pred_list is None:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().numpy(), axis=0)

        return self._full_sort_metrics(epoch, answer_list, pred_list)
