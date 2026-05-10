from collections import defaultdict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Tuple, Callable, Literal

import torch
from lightning import LightningDataModule
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision.transforms.v2 import (
    Compose,
    RandomHorizontalFlip,
    ColorJitter,
    RandomRotation,
)
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pixeltable as pxt
from goldener.split import GoldSplitter

from image_classification_cifar10.utils import (
    get_gold_splitter,
    get_gold_descriptor,
    get_gold_batcher,
    CIFAR10_PREPROCESS,
)

logger = getLogger(__name__)


class GoldCifar10(CIFAR10):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: None | Callable = None,
        target_transform: None | Callable = None,
        download: bool = False,
        count: int | None = None,
        remove_ratio: float | None = None,
        duplicate_table_path: str | None = None,
        drop_duplicate_table: bool = True,
        to_duplicate_clusters: int | None = None,
        cluster_count: int | None = None,
        duplicate_per_sample: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.count = count
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # keep only a subset if count is specified
        # only the first 'count' samples are kept
        if count is not None:
            self.data: np.ndarray = self.data[:count]
            self.targets: list[int] = self.targets[:count]

        # keep only a subset if remove_ratio is specified
        # The removal is done randomly and stratified by class labels
        if remove_ratio is not None:
            training_indices, excluded = train_test_split(
                range(len(self)),
                test_size=remove_ratio,
                random_state=random_state,
                shuffle=True,
                stratify=self.targets_as_array,
            )
            self.data = self.data[training_indices]
            self.targets = [self.targets[i] for i in training_indices]
            self.excluded_indices = excluded
        else:
            self.excluded_indices = []

        # duplicate samples based on clustering if all duplication parameters are specified
        # every new data added from duplication is an augmented version of the initial data
        self.duplicated_indices: list[int] = []
        duplication_params = (
            duplicate_table_path,
            to_duplicate_clusters,
            cluster_count,
            duplicate_per_sample,
        )
        if any(duplication_params):
            if not all(duplication_params):
                raise ValueError(
                    "If any duplication parameter is set, all must be set."
                )

            # extract the features using goldener
            assert duplicate_table_path is not None
            gold_descriptor = get_gold_descriptor(
                table_name=duplicate_table_path,
                min_pxt_insert_size=10000,
                batch_size=16,
                num_workers=16,
                to_keep_schema={"label": pxt.String},
            )
            if drop_duplicate_table:
                pxt.drop_table(duplicate_table_path, if_not_exists="ignore")

            with torch.no_grad():
                vectorized = gold_descriptor.describe_in_table(self)
            torch.cuda.empty_cache()

            # group features and specific indices by label
            features_per_label = defaultdict(list)
            indices_per_label = defaultdict(list)
            for row in vectorized.select(
                vectorized.idx, vectorized.embeddings, vectorized.label
            ).collect():
                features_per_label[row["label"]].append(row["embeddings"])
                indices_per_label[row["label"]].append(row["idx"])

            # perform clustering and duplication per label
            random_generator = np.random.default_rng(random_state)
            duplication_transform = Compose(
                [
                    RandomHorizontalFlip(),
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    RandomRotation(degrees=15),
                ]
            )
            for label, features in features_per_label.items():
                assert cluster_count is not None and duplicate_per_sample is not None
                label_indices = indices_per_label[label]
                logger.info(f"Adding duplicates for label {label}")
                kmeans = KMeans(
                    n_clusters=cluster_count,
                    random_state=random_state,
                    n_init="auto",
                ).fit(np.stack(features, axis=0))
                cluster_indices = random_generator.choice(
                    range(cluster_count),
                    size=to_duplicate_clusters,
                    replace=False,
                )
                logger.info(f"The selected clusters are {cluster_indices}")

                for data_idx, cluster_id in enumerate(kmeans.labels_):
                    if cluster_id in cluster_indices:
                        duplicated_idx = label_indices[data_idx]
                        to_add_data = np.vstack(
                            [
                                duplication_transform(
                                    self.data[duplicated_idx][np.newaxis, ...]
                                )
                                for _ in range(duplicate_per_sample)
                            ]
                        )
                        self.data = np.vstack([self.data, to_add_data])
                        self.targets.extend(
                            [self.targets[duplicated_idx]] * duplicate_per_sample
                        )
                        self.duplicated_indices.append(duplicated_idx)
        else:
            self.duplicated_indices = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        if self.count is not None and index >= self.count:
            raise IndexError("Index out of range for GoldCifar10 with limited count.")
        return super().__getitem__(index) + (index,)

    @property
    def targets_as_array(self) -> np.ndarray:
        return np.array(self.targets)


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.data_dir = cfg.data.cache
        self.goldener_config = cfg.goldener_config

        self.random_state = cfg.exp.random_state

        self.val_ratio = cfg.exp.val_ratio
        self.split_method = cfg.exp.split_method

        self.random_split_state = cfg.data.random_split_state
        self.random_shuffle_state = cfg.data.random_shuffle_state
        self.remove_ratio = cfg.data.remove_ratio
        self.duplicate_table_path = cfg.data.duplicate_table_path
        self.drop_duplicate_table = cfg.data.drop_duplicate_table
        self.to_duplicate_clusters = cfg.data.to_duplicate_clusters
        self.cluster_count = cfg.data.cluster_count
        self.duplicate_per_sample = cfg.data.duplicate_per_sample

        self.batch_size = cfg.exp.batch_size
        self.num_workers = cfg.data.num_workers
        self.max_batches = cfg.debug_count.train_count
        self.train_count = (
            self.max_batches * self.batch_size if self.max_batches is not None else None
        )
        self.validate_on_test = cfg.exp.validate_on_test

        # Define transforms
        self.transform = CIFAR10_PREPROCESS
        self.train_transforms = Compose(
            [
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                RandomRotation(degrees=15),
            ]
            + self.transform.transforms
        )

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

        self.excluded_train_indices: Subset
        self.duplicated_train_indices: list[int]

        self.gold_train_indices: list[int]
        self.gold_val_indices: list[int]
        self.gold_train_dataset: Subset
        self.gold_val_dataset: Subset

        self.sk_train_indices: list[int]
        self.sk_val_indices: list[int]
        self.sk_train_dataset: Subset
        self.sk_val_dataset: Subset

        self.test_dataset: GoldCifar10

    @property
    def settings_as_str(self) -> str:
        return (
            f"settings_{self.random_state}_{self.remove_ratio}"
            f"_{self.cluster_count}_{self.to_duplicate_clusters}"
            f"_{self.duplicate_per_sample}"
        ).replace(".", "_")

    def prepare_data(self) -> None:
        # Download CIFAR-10
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            val_dataset = GoldCifar10(
                root=self.data_dir,
                train=True,
                transform=self.transform,
                download=False,
                count=self.train_count,
                random_state=self.random_state,
                remove_ratio=self.remove_ratio,
                duplicate_table_path=(
                    f"{self.duplicate_table_path}_{self.settings_as_str}"
                    if self.duplicate_table_path is not None
                    else None
                ),
                drop_duplicate_table=self.drop_duplicate_table,
                to_duplicate_clusters=self.to_duplicate_clusters,
                cluster_count=self.cluster_count,
                duplicate_per_sample=self.duplicate_per_sample,
            )
            # the dataset used to train the model will benefit from data augmentation
            train_dataset = deepcopy(val_dataset)
            train_dataset.transform = self.train_transforms
            self.duplicated_train_indices = val_dataset.duplicated_indices
            self.excluded_train_indices = val_dataset.excluded_indices

            # make random splitting with sklearn
            if self.split_method in ("random", "all"):
                self.sk_train_indices, self.sk_val_indices = train_test_split(
                    range(len(val_dataset)),
                    test_size=int(self.val_ratio * len(val_dataset)),
                    random_state=self.random_split_state,
                    shuffle=True,
                    stratify=val_dataset.targets_as_array,
                )
                self.sk_train_dataset = Subset(train_dataset, self.sk_train_indices)
                self.sk_val_dataset = Subset(val_dataset, self.sk_val_indices)

            # make gold splitting
            if self.split_method in ("gold", "all"):
                with torch.no_grad():
                    split_table = self.gold_splitter.split_in_table(val_dataset)
                splits = self.gold_splitter.get_split_indices(
                    split_table, selection_key="selected", idx_key="idx"
                )

                self.gold_train_indices = list(splits["train"])
                self.gold_val_indices = list(splits["val"])
                self.gold_train_dataset = Subset(train_dataset, self.gold_train_indices)
                self.gold_val_dataset = Subset(val_dataset, self.gold_val_indices)

        if stage == "test" or stage is None:
            self.test_dataset = GoldCifar10(
                root=self.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )

    def _get_features_by_indices(
        self,
        indices: list[int],
        label: str | None = None,
    ) -> list[np.ndarray]:
        vectorized = pxt.get_table(self.gold_splitter.descriptor.table_path)
        assert vectorized is not None
        query = vectorized.idx.isin(indices)
        if label is not None:
            query = query & (vectorized.label == label)  # type: ignore[assignment]

        return [
            row["features"]
            for row in vectorized.where(query).select(vectorized.features).collect()
        ]

    def get_gold_train_features(self, label: str | None = None) -> list[np.ndarray]:
        return self._get_features_by_indices(
            self.gold_train_indices,
            label,
        )

    def get_gold_val_features(self, label: str | None = None) -> list[np.ndarray]:
        return self._get_features_by_indices(
            self.gold_val_indices,
            label,
        )

    def get_sk_train_features(self, label: str | None = None) -> list[np.ndarray]:
        return self._get_features_by_indices(
            self.sk_train_indices,
            label,
        )

    def get_sk_val_features(self, label: str | None = None) -> list[np.ndarray]:
        return self._get_features_by_indices(
            self.sk_val_indices,
            label,
        )

    def _get_batch_args(
        self,
        batch_method: Literal["gold", "random"],
        dataset: Subset,
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
            with torch.no_grad():
                return {
                    "batch_sampler": get_gold_batcher(
                        dataset=dataset,
                        goldener_config=self.goldener_config,
                        name_prefix=self.settings_as_str,
                        batch_size=self.batch_size,
                        generator=generator,
                        max_batches=self.max_batches,
                        update_batch=self.goldener_config.update_batch,
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
