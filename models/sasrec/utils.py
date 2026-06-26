from __future__ import annotations

import json
import os
import random

import numpy as np
import scipy.sparse as sp
import torch

try:
    from accelerate.utils import set_seed as accelerate_set_seed
except ImportError:  # pragma: no cover - accelerate is present in the target env.
    accelerate_set_seed = None


def set_seed(seed: int, reproducibility: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if accelerate_set_seed is not None:
        accelerate_set_seed(seed)
    torch.backends.cudnn.deterministic = reproducibility
    torch.backends.cudnn.benchmark = not reproducibility


def check_path(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def neg_sample(item_set: set[int], item_size: int) -> int:
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    def __init__(self, checkpoint_path: str, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def compare(self, score: np.ndarray) -> bool:
        return bool(np.all(score <= self.best_score + self.delta))

    def __call__(self, score: np.ndarray, model: torch.nn.Module) -> None:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            return

        if self.compare(score):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.save_checkpoint(score, model)
        self.counter = 0

    def save_checkpoint(self, score: np.ndarray, model: torch.nn.Module) -> None:
        if self.verbose:
            print(f"Validation score improved to {score.tolist()}; saving to {self.checkpoint_path}")
        torch.save(model.state_dict(), self.checkpoint_path)


def generate_rating_matrix_valid(user_seq: list[list[int]], num_users: int, num_items: int):
    row, col, data = [], [], []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:
            row.append(user_id)
            col.append(item)
            data.append(1)
    return sp.csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(num_users, num_items))


def generate_rating_matrix_test(user_seq: list[list[int]], num_users: int, num_items: int):
    row, col, data = [], [], []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:
            row.append(user_id)
            col.append(item)
            data.append(1)
    return sp.csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(num_users, num_items))


def get_user_seqs(data_file: str):
    with open(data_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    user_seq: list[list[int]] = []
    item_set: set[int] = set()
    for line in lines:
        _, items = line.strip().split(" ", 1)
        item_ids = [int(item) for item in items.split(" ")]
        user_seq.append(item_ids)
        item_set |= set(item_ids)

    max_item = max(item_set)
    num_users = len(lines)
    num_items = max_item + 2
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix


def get_metric(pred_list: np.ndarray, topk: int = 10):
    ndcg = 0.0
    hit = 0.0
    mrr = 0.0
    for rank in pred_list:
        mrr += 1.0 / (rank + 1.0)
        if rank < topk:
            ndcg += 1.0 / np.log2(rank + 2.0)
            hit += 1.0
    return hit / len(pred_list), ndcg / len(pred_list), mrr / len(pred_list)


def recall_at_k(actual: np.ndarray, predicted: np.ndarray, topk: int) -> float:
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for idx in range(num_users):
        actual_set = set(actual[idx])
        pred_set = set(predicted[idx][:topk])
        if not actual_set:
            continue
        sum_recall += len(actual_set & pred_set) / float(len(actual_set))
        true_users += 1
    return sum_recall / true_users


def idcg_k(k: int) -> float:
    value = 0.0
    for i in range(k):
        value += 1.0 / np.log2(i + 2)
    return value if value else 1.0


def ndcg_k(actual: np.ndarray, predicted: np.ndarray, topk: int) -> float:
    res = 0.0
    for user_id in range(len(actual)):
        actual_set = set(actual[user_id])
        dcg_k = 0.0
        for j in range(topk):
            if predicted[user_id][j] in actual_set:
                dcg_k += 1.0 / np.log2(j + 2)
        res += dcg_k / idcg_k(min(topk, len(actual_set)))
    return res / float(len(actual))


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
