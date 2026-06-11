from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from .utils import neg_sample


class SASRecModernizedDataset(Dataset):
    def __init__(self, args, user_seq: Sequence[Sequence[int]], data_type: str = "train"):
        self.args = args
        self.user_seq = list(user_seq)
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __len__(self) -> int:
        return len(self.user_seq)

    def __getitem__(self, index: int):
        user_id = index
        items = list(self.user_seq[index])

        if self.data_type not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )
