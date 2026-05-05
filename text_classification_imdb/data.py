from logging import getLogger
from typing import Literal

import torch
from datasets import load_dataset
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader
import pixeltable as pxt
from transformers import AutoTokenizer
from goldener.split import GoldSplitter

from text_classification_imdb.utils import get_gold_splitter, get_gold_batcher

logger = getLogger(__name__)


class IMDbDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256,
        count: int | None = None,
        remove_ratio: float | None = None,
        random_state: int = 42,
    ) -> None:
        raw = load_dataset("imdb", split=split)
        if count is not None:
            raw = raw.select(range(count))
        self._data = raw

        if remove_ratio is not None:
            training_indices, excluded = train_test_split(
                range(len(self)),
                test_size=remove_ratio,
                random_state=random_state,
                shuffle=True,
                stratify=self.targets_as_array,
            )
            self._data = self._data.select(training_indices)
            self.excluded_indices = excluded
        else:
            self.excluded_indices = []

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        item = self._data[index]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label: int = item["label"]
        return input_ids, attention_mask, label, index

    @property
    def targets(self) -> list[int]:
        return [int(item["label"]) for item in self._data]

    @property
    def targets_as_array(self):
        import numpy as np

        return np.array(self.targets)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


class IMDbDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.tokenizer_name: str = cfg.data.tokenizer_name
        self.max_length: int = cfg.data.max_length
        self.goldener_config = cfg.goldener_config

        self.random_state: int = cfg.exp.random_state
        self.val_ratio: float = cfg.exp.val_ratio
        self.split_method = cfg.exp.split_method
        self.random_split_state: int = cfg.data.random_split_state
        self.random_shuffle_state: int = cfg.data.random_shuffle_state
        self.remove_ratio = cfg.data.remove_ratio

        self.batch_size: int = cfg.exp.batch_size
        self.num_workers: int = cfg.data.num_workers

        self.max_batches = cfg.debug_count.train_count
        self.train_count: int | None = (
            self.max_batches * self.batch_size if self.max_batches is not None else None
        )
        self.test_count: int | None = (
            cfg.debug_count.test_count * self.batch_size
            if cfg.debug_count.test_count is not None
            else None
        )
        self.validate_on_test: bool = cfg.exp.validate_on_test

        self.gold_splitter: GoldSplitter = get_gold_splitter(
            goldener_config=self.goldener_config,
            name_prefix=self.settings_as_str,
            val_ratio=self.val_ratio,
            max_batches=self.max_batches,
        )
        if cfg.goldener_config.update_selection:
            pxt.drop_table(
                self.gold_splitter.selector.table_path, if_not_exists="ignore"
            )
            pxt.drop_table(
                self.gold_splitter.descriptor.table_path, if_not_exists="ignore"
            )

        self.gold_train_indices: list[int]
        self.gold_val_indices: list[int]
        self.gold_train_dataset: Subset
        self.gold_val_dataset: Subset

        self.sk_train_indices: list[int]
        self.sk_val_indices: list[int]
        self.sk_train_dataset: Subset
        self.sk_val_dataset: Subset

        self.test_dataset: IMDbDataset

    @property
    def settings_as_str(self) -> str:
        return (
            f"imdb_{self.tokenizer_name.replace('-', '_')}_{self.max_length}"
            f"_{self.random_state}_{self.remove_ratio}"
        ).replace(".", "_")

    @property
    def vocab_size(self) -> int:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_name).vocab_size

    def prepare_data(self) -> None:
        load_dataset("imdb")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            training_dataset = IMDbDataset(
                split="train",
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                count=self.train_count,
                remove_ratio=self.remove_ratio,
            )
            self.excluded_train_indices = training_dataset.excluded_indices

            # Random splitting with sklearn (stratified)
            if self.split_method in ("random", "all"):
                self.sk_train_indices, self.sk_val_indices = train_test_split(
                    range(len(training_dataset)),
                    test_size=int(self.val_ratio * len(training_dataset)),
                    random_state=self.random_split_state,
                    shuffle=True,
                    stratify=training_dataset.targets_as_array,
                )
                self.sk_train_dataset = Subset(training_dataset, self.sk_train_indices)
                self.sk_val_dataset = Subset(training_dataset, self.sk_val_indices)

            # Smart splitting with GoldSplitter
            if self.split_method in ("gold", "all"):
                with torch.no_grad():
                    split_table = self.gold_splitter.split_in_table(training_dataset)
                splits = self.gold_splitter.get_split_indices(
                    split_table, selection_key="selected", idx_key="idx"
                )
                self.gold_train_indices = list(splits["train"])
                self.gold_val_indices = list(splits["val"])
                self.gold_train_dataset = Subset(
                    training_dataset, self.gold_train_indices
                )
                self.gold_val_dataset = Subset(training_dataset, self.gold_val_indices)

        if stage == "test" or stage is None:
            self.test_dataset = IMDbDataset(
                split="test",
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                count=self.test_count,
            )

    def _get_batch_args(
        self,
        batch_method: Literal["gold", "random"],
        dataset: Dataset,
    ) -> dict:
        generator = torch.Generator().manual_seed(self.random_shuffle_state)
        if batch_method == "random":
            return {
                "batch_size": self.batch_size,
                "shuffle": True,
                "generator": generator,
                "drop_last": True,
            }
        else:
            return {
                "batch_sampler": get_gold_batcher(
                    dataset=dataset,
                    goldener_config=self.goldener_config,
                    name_prefix=self.settings_as_str,
                    batch_size=self.batch_size,
                    generator=generator,
                    max_batches=self.max_batches,
                )
            }

    def sk_train_dataloader(
        self, batch_method: Literal["gold", "random"]
    ) -> DataLoader:
        return DataLoader(
            dataset=self.sk_train_dataset,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            **self._get_batch_args(
                batch_method=batch_method,
                dataset=self.sk_train_dataset,
            ),
        )

    def sk_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.sk_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def gold_train_dataloader(
        self, batch_method: Literal["gold", "random"]
    ) -> DataLoader:
        return DataLoader(
            self.gold_train_dataset,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            **self._get_batch_args(
                batch_method=batch_method,
                dataset=self.gold_train_dataset,
            ),
        )

    def gold_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.gold_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
