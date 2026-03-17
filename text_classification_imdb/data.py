from logging import getLogger

import torch
from datasets import load_dataset
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader
import pixeltable as pxt
from transformers import AutoTokenizer
from goldener.split import GoldSplitter

from text_classification_imdb.utils import get_gold_splitter

logger = getLogger(__name__)


class IMDbDataset(Dataset):
    """IMDb Movie Reviews dataset with WordPiece tokenization.

    Args:
        split: HuggingFace dataset split – ``"train"`` or ``"test"``.
        tokenizer_name: Name of the pretrained tokenizer (WordPiece).
        max_length: Maximum token sequence length (shorter texts are padded).
        count: Optional limit on the number of samples (for debugging).
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256,
        count: int | None = None,
    ) -> None:
        raw = load_dataset("imdb", split=split)
        if count is not None:
            raw = raw.select(range(count))
        self._data = raw
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

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
        """Return all labels as a list (for stratified splitting)."""
        return [int(item["label"]) for item in self._data]

    @property
    def targets_as_array(self):
        import numpy as np

        return np.array(self.targets)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


class IMDbDataModule(LightningDataModule):
    """Lightning DataModule for IMDb sentiment classification.

    Prepares two split strategies:
    - **Random split** via scikit-learn :func:`train_test_split`.
    - **Smart split** via GoldSplitter from the Goldener library.

    Args:
        cfg: Hydra DictConfig with keys ``data``, ``exp``, ``gold_splitter``,
            ``logging``, and ``debug_count``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.tokenizer_name: str = cfg.data.tokenizer_name
        self.max_length: int = cfg.data.max_length
        self.gold_splitter_cfg = cfg.gold_splitter

        self.random_state: int = cfg.exp.random_state
        self.val_ratio: float = cfg.exp.val_ratio
        self.random_split_state: int = cfg.data.random_split_state

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
            splitter_cfg=self.gold_splitter_cfg,
            name_prefix=self.settings_as_str,
            val_ratio=self.val_ratio,
            max_batches=self.max_batches,
        )
        if cfg.gold_splitter.update_selection:
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def settings_as_str(self) -> str:
        return (
            f"imdb_{self.tokenizer_name.replace('-', '_')}_{self.max_length}"
            f"_{self.random_state}"
        )

    @property
    def vocab_size(self) -> int:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_name).vocab_size

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download the IMDb dataset (cached by HuggingFace datasets)."""
        load_dataset("imdb")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full_dataset = IMDbDataset(
                split="train",
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                count=self.train_count,
            )

            # Random splitting with sklearn (stratified)
            self.sk_train_indices, self.sk_val_indices = train_test_split(
                range(len(full_dataset)),
                test_size=int(self.val_ratio * len(full_dataset)),
                random_state=self.random_split_state,
                shuffle=True,
                stratify=full_dataset.targets_as_array,
            )
            self.sk_train_dataset = Subset(full_dataset, self.sk_train_indices)
            self.sk_val_dataset = Subset(full_dataset, self.sk_val_indices)

            # Smart splitting with GoldSplitter
            split_table = self.gold_splitter.split_in_table(full_dataset)
            splits = self.gold_splitter.get_split_indices(
                split_table, selection_key="selected", idx_key="idx"
            )
            self.gold_train_indices = list(splits["train"])
            self.gold_val_indices = list(splits["val"])
            self.gold_train_dataset = Subset(full_dataset, self.gold_train_indices)
            self.gold_val_dataset = Subset(full_dataset, self.gold_val_indices)

        if stage == "test" or stage is None:
            self.test_dataset = IMDbDataset(
                split="test",
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                count=self.test_count,
            )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def sk_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.sk_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.random_state),
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

    def gold_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.gold_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.random_state),
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
