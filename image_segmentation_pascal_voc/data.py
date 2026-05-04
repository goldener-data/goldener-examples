from collections import defaultdict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Callable
from tqdm import tqdm

import torch
from lightning import LightningDataModule
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Subset, DataLoader
from any_gold import PascalVOC2012Segmentation
import torchvision
from torchvision.transforms.v2 import (
    Compose,
    ColorJitter,
    Resize,
    ToImage,
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pixeltable as pxt
from goldener.split import GoldSplitter
from image_segmentation_pascal_voc.utils import (
    get_gold_splitter,
    get_gold_descriptor,
    PASCAL_VOC_PREPROCESS,
    collate_pascal_voc,
    multilabel_iterative_train_test_split,
)

logger = getLogger(__name__)


class GoldPascalVOC2012Segmentation(PascalVOC2012Segmentation):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: None | Callable = None,
        target_transform: None | Callable = None,
        override: bool = False,
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
            split=split,
            transform=transform,
            target_transform=target_transform,
            override=override,
        )

        # keep only a subset if count is specified
        # only the first 'count' samples are kept
        original_length = len(self)
        if count is not None and count < original_length:
            self.samples: list[Path] = self.samples[:count]

        # keep only a subset if remove_ratio is specified
        # The removal is done randomly
        if remove_ratio is not None:
            multilabel_iterative_train_test_split(
                self.get_index_labels(32, 8),
                test_size=remove_ratio,
                random_state=random_state,
            )
            training_indices, excluded = train_test_split(
                range(len(self.samples)),
                test_size=remove_ratio,
                random_state=random_state,
                shuffle=True,
            )
            self.samples = [self.samples[i] for i in training_indices]
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
                min_pxt_insert_size=1000,
                batch_size=32,
                num_workers=8,
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
                        # For VOC, we need to duplicate both image and mask paths
                        for _ in range(duplicate_per_sample):
                            self.samples.append(self.samples[duplicated_idx])
                        self.duplicated_indices.append(duplicated_idx)
        else:
            self.duplicated_indices = []

    def get_index_labels(
        self,
        batch_size: int = 32,
        num_workers: int = 8,
    ) -> dict[int, set[str]]:
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda batch_list: batch_list,
            drop_last=False,
        )

        index_label = {}

        for batch in tqdm(dataloader, desc="Getting labels per index"):
            for sample in batch:
                index_label[sample["index"]] = set(
                    [
                        label
                        for label in sample["labels"]
                        if label not in ("void", "background")
                    ]
                )

        return index_label


class VOCSegmentationDataModule(LightningDataModule):
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
        self.transform = PASCAL_VOC_PREPROCESS
        self.train_transforms = Compose(
            [
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
            + self.transform.transforms
        )
        # For masks, we only need to convert to tensor and resize
        # No normalization should be applied to segmentation masks
        self.mask_transform = Compose(
            [
                ToImage(),
                Resize(
                    (224, 224),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
            ]
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

        self.test_dataset: PascalVOC2012Segmentation

    @property
    def settings_as_str(self) -> str:
        return (
            f"settings_{self.random_state}_{self.remove_ratio}"
            f"_{self.cluster_count}_{self.to_duplicate_clusters}"
            f"_{self.duplicate_per_sample}"
        ).replace(".", "_")

    def prepare_data(self) -> None:
        # Download Pascal VOC
        PascalVOC2012Segmentation(root=self.data_dir, split="train", override=False)
        PascalVOC2012Segmentation(root=self.data_dir, split="val", override=False)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            val_dataset = GoldPascalVOC2012Segmentation(
                root=self.data_dir,
                split="train",
                transform=PASCAL_VOC_PREPROCESS,
                target_transform=self.mask_transform,
                override=False,
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
            self.duplicated_train_indices = val_dataset.duplicated_indices
            self.excluded_train_indices = val_dataset.excluded_indices

            # make gold splitting
            if self.split_method in ("gold", "all"):
                with torch.no_grad():
                    split_table = self.gold_splitter.split_in_table(val_dataset)
                splits = self.gold_splitter.get_split_indices(
                    split_table, selection_key="selected", idx_key="idx"
                )
                self.gold_train_indices = list(splits["train"])
                self.gold_val_indices = list(splits["val"])

            # make random splitting with sklearn
            if self.split_method in ("random", "all"):
                (
                    self.sk_train_indices,
                    self.sk_val_indices,
                ) = multilabel_iterative_train_test_split(
                    val_dataset.get_index_labels(self.batch_size, self.num_workers),
                    test_size=self.val_ratio,
                    random_state=self.random_split_state,
                )

            # assign datasets
            val_dataset.target_transform = self.mask_transform
            train_dataset = deepcopy(val_dataset)
            train_dataset.transform = self.train_transforms

            if self.split_method in ("gold", "all"):
                self.gold_train_dataset = Subset(train_dataset, self.gold_train_indices)
                self.gold_val_dataset = Subset(val_dataset, self.gold_val_indices)

            if self.split_method in ("random", "all"):
                self.sk_train_dataset = Subset(train_dataset, self.sk_train_indices)
                self.sk_val_dataset = Subset(val_dataset, self.sk_val_indices)

        if stage == "test" or stage is None:
            self.test_dataset = GoldPascalVOC2012Segmentation(
                root=self.data_dir,
                split="val",
                transform=self.transform,
                target_transform=self.mask_transform,
                override=False,
            )

    def sk_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.sk_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.random_shuffle_state),
            collate_fn=collate_pascal_voc,
            drop_last=True,
        )

    def sk_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.sk_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_pascal_voc,
            drop_last=True,
        )

    def gold_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.gold_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.random_shuffle_state),
            collate_fn=collate_pascal_voc,
            drop_last=True,
        )

    def gold_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.gold_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_pascal_voc,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=collate_pascal_voc,
        )
