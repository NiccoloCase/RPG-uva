from __future__ import annotations

import sys
from pathlib import Path


def _ensure_third_party_on_path() -> None:
    """Expose the vendored `genrec` package without editing the submodule."""
    repo_root = Path(__file__).resolve().parents[2]
    third_party_root = repo_root / "third_party"
    if str(third_party_root) not in sys.path:
        sys.path.insert(0, str(third_party_root))


_ensure_third_party_on_path()

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class SASRecTokenizer(AbstractTokenizer):
    """Repo-owned SASRec tokenizer scaffold.

    Standard SASRec uses plain item IDs as tokens:
    - `0` is padding.
    - every real item is represented by its dataset item ID.
    - there is no semantic-code tokenizer and usually no EOS token.

    The typical implementation clones three behaviors:
    1. truncate to the most recent `max_item_seq_len` interactions,
    2. pad shorter sequences with `0`,
    3. produce next-item supervision for either every timestep or the last
       timestep, depending on the training recipe you choose.
    """

    def __init__(self, config: dict, dataset: AbstractDataset):
        super().__init__(config, dataset)
        self.dataset = dataset
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping["id2item"]
        self.ignored_label = -100
        self.eos_token = None

    @property
    def vocab_size(self) -> int:
        """SASRec operates directly over item IDs, including padding ID 0."""
        return self.dataset.n_items

    @property
    def max_token_seq_len(self) -> int:
        return self.config["max_item_seq_len"]

    def _pad_input_ids(self, input_ids: list[int]) -> tuple[list[int], list[int], int]:
        """Right-pad a truncated sequence and build its attention mask."""
        input_ids = input_ids[-self.max_token_seq_len:]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        pad_len = self.max_token_seq_len - seq_lens

        input_ids = input_ids + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        return input_ids, attention_mask, seq_lens

    def _tokenize_train_sequence(self, item_seq: list[int]) -> dict:
        """Create sliding-window training examples with one next-item label each."""
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []

        for target_idx in range(1, len(item_seq)):
            input_ids = item_seq[max(0, target_idx - self.max_token_seq_len):target_idx]
            padded_input_ids, attention_mask, seq_lens = self._pad_input_ids(input_ids)
            all_input_ids.append(padded_input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append([item_seq[target_idx]])
            all_seq_lens.append(seq_lens)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
            "seq_lens": all_seq_lens,
        }

    def _tokenize_eval_sequence(self, item_seq: list[int]) -> dict:
        """Create one evaluation example that predicts the final held-out item."""
        input_ids = item_seq[:-1]
        padded_input_ids, attention_mask, seq_lens = self._pad_input_ids(input_ids)

        return {
            "input_ids": [padded_input_ids],
            "labels": [[item_seq[-1]]],
            "attention_mask": [attention_mask],
            "seq_lens": [seq_lens],
        }

    def tokenize_function(self, example: dict, split: str) -> dict:
        """Tokenize one dataset row into SASRec-style examples."""
        item_seq = [self.item2id[item] for item in example["item_seq"][0]]

        if split == "train":
            return self._tokenize_train_sequence(item_seq)

        return self._tokenize_eval_sequence(item_seq)

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}

        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config["num_proc"],
                desc=f"Tokenizing {split} set: ",
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type="torch")

        return tokenized_datasets
