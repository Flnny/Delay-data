"""Batched data loading for mambular (speed only — outputs are unchanged).

mambular's MambularDataset.__getitem__ builds one tensor per feature per ROW and
the DataLoader then collates 2048 rows one by one. With ~20-50 PLE-expanded
features this runs at ~2 batches/s on CPU and dominates both training and
inference time.

Here the DataLoader is given a BatchSampler and batch_size=None, so __getitem__
receives the whole list of indices and tensor fancy-indexing returns the batch in
one call. This is exactly the sampler stack DataLoader builds internally for
`shuffle=True/False` (BatchSampler(RandomSampler|SequentialSampler, drop_last=False)),
so the index order — and therefore every batch — is identical; only the fetch is
vectorised. common/verify_fast_loader.py checks bit-identical predictions.
"""
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

_PATCHED = False


def _loader(dataset, batch_size, shuffle, kwargs):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(sampler, batch_size, drop_last=False),
                      batch_size=None, **kwargs)


def _get(self, name):
    if not hasattr(self, name):
        raise ValueError(f"No {name.replace('_dataset', '')} dataset provided!")
    return getattr(self, name)


def enable() -> None:
    """Monkeypatch MambularDataModule's four *_dataloader methods (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    from mambular.data_utils.datamodule import MambularDataModule as DM

    DM.train_dataloader = lambda self: _loader(_get(self, "train_dataset"), self.batch_size,
                                               self.shuffle, self.dataloader_kwargs)
    DM.val_dataloader = lambda self: _loader(_get(self, "val_dataset"), self.batch_size,
                                             False, self.dataloader_kwargs)
    DM.test_dataloader = lambda self: _loader(_get(self, "test_dataset"), self.batch_size,
                                              False, self.dataloader_kwargs)
    DM.predict_dataloader = lambda self: _loader(_get(self, "predict_dataset"), self.batch_size,
                                                 False, self.dataloader_kwargs)
    _PATCHED = True
