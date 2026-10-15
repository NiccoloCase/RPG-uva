# MemGen-GR configs

One file per model and dataset, loaded by `scripts/memgen_train.py` on top of the
MemGen-GR defaults (`third_party/MemGen-GR/genrec/default.yaml`, the dataset
config and the model config). A file holds only the Amazon category and, for
Books, the batch sizes. Paths, seeds and logging are set by the launcher.
