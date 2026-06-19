from __future__ import annotations

from genrec.models.RPG.tokenizer import RPGTokenizer


class DRPGTokenizer(RPGTokenizer):
    """RPG semantic-ID tokenizer with DRPG history/target fields.

    The inherited RPG tokenizer owns sentence embedding, OPQ/PQ semantic-ID
    generation, and item-token mapping. DRPG only changes the examples consumed
    by the model: histories are encoded as semantic-ID matrices and targets are
    the next item's full semantic ID.
    """

    def tokenize_function(self, example: dict, split: str) -> dict:
        max_history_len = self.config.get("max_history_len", self.config["max_item_seq_len"])
        item_seq = example["item_seq"][0]

        if split == "train":
            all_history_sid = []
            all_history_mask = []
            all_decoder_labels = []

            # Start from the second item so every target has at least one
            # historical item to condition on.
            for index in range(1, len(item_seq)):
                history_items = item_seq[max(0, index - max_history_len):index]
                target_item = item_seq[index]

                history_sid, history_mask = self.encode_history(history_items, max_history_len)
                decoder_labels = self.encode_target(target_item)

                all_history_sid.append(history_sid)
                all_history_mask.append(history_mask)
                all_decoder_labels.append(decoder_labels)

            return {
                "history_sid": all_history_sid,
                "history_mask": all_history_mask,
                "decoder_labels": all_decoder_labels,
            }

        history_items = item_seq[:-1][-max_history_len:]
        target_item = item_seq[-1]
        history_sid, history_mask = self.encode_history(history_items, max_history_len)

        return {
            "history_sid": [history_sid],
            "history_mask": [history_mask],
            "labels": [[self.item2id[target_item]]],
        }

    def encode_history(self, history_items: list, max_history_len: int) -> tuple[list[list[int]], list[bool]]:
        history_sid = []
        history_mask = []

        for item in history_items:
            if item in self.item2tokens:
                history_sid.append(list(self.item2tokens[item]))
                history_mask.append(True)
            else:
                history_sid.append([0] * self.n_digit)
                history_mask.append(False)

        while len(history_sid) < max_history_len:
            history_sid.append([0] * self.n_digit)
            history_mask.append(False)

        return history_sid, history_mask

    def encode_target(self, target_item) -> list[int]:
        if target_item in self.item2tokens:
            return list(self.item2tokens[target_item])
        return [self.ignored_label] * self.n_digit
