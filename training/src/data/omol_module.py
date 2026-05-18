import logging
import random

import lightning as L
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


class OMolModule(L.LightningDataModule):
    """LightningDataModule for the OMol force-field dataset."""

    def __init__(self, data, batch_size):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage=None):
        train_dataset = AseDBDataset(
            {"src": self.hparams.data.train_data_path, "a2g_args": dict(r_energy=True, r_forces=True), "keep_in_memory": True}
        )

        if self.hparams.data.val_data_path is None:
            train_dataset, val_dataset = self.train_val_split(
                dataset=train_dataset,
                val_ratio=self.hparams.data.val_ratio,
                shuffle=self.hparams.data.shuffle,
                seed=self.hparams.data.seed,
            )
        else:
            val_dataset = AseDBDataset({"src": self.hparams.data.val_data_path, "a2g_args": dict(r_energy=True, r_forces=True)})

        self.datasets = {
            "train": train_dataset,
            "val": val_dataset,
        }

        log.info(f"# Training: {len(self.datasets['train'])}")
        log.info(f"# Validation: {len(self.datasets['val'])}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size.train,
            shuffle=self.hparams.data.shuffle,
            num_workers=self.hparams.data.num_workers,
            pin_memory=self.hparams.data.pin_memory,
            collate_fn=atomicdata_list_to_batch,
            persistent_workers=self.hparams.data.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size.val,
            shuffle=False,
            num_workers=self.hparams.data.num_workers,
            pin_memory=self.hparams.data.pin_memory,
            collate_fn=atomicdata_list_to_batch,
            persistent_workers=self.hparams.data.num_workers > 0,
        )
