import pixeltable as pxt

import hydra
import timm
import torch
from PIL.Image import Image
from sklearn.cluster import KMeans
from goldener.vision.vectorizers import get_vit_class_token_vectorizer
from goldener import (
    GoldSKLearnClusteringTool,
    GoldClusterizer,
    GoldDescriptor,
    GoldTorchEmbeddingTool,
    GoldTorchEmbeddingToolConfig,
    GoldGreedyKCenterSelectionTool,
    GoldSelector,
    GoldSet,
    GoldSplitter,
)

from omegaconf import DictConfig
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, Resize

CIFAR10_PREPROCESS = Compose(
    [
        ToTensor(),
        Resize(224),
        Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
    ]
)


def collate_cifar10(
    batch: list[tuple[Image, int, int]],
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    images, targets, indices = zip(*batch)
    imgs_tensor = torch.stack([CIFAR10_PREPROCESS(image) for image in images])
    str_targets = [str(target) for target in targets]
    idx_list = [int(idx) for idx in indices]
    return {
        "data": imgs_tensor,
        "label": str_targets,
        "idx": idx_list,
    }


def get_gold_descriptor(
    table_name: str,
    min_pxt_insert_size: int,
    batch_size: int,
    num_workers: int,
    to_keep_schema: dict,
) -> GoldDescriptor:
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )

    embedder = GoldTorchEmbeddingTool(
        GoldTorchEmbeddingToolConfig(
            model=timm.create_model(
                model_name="vit_large_patch16_dinov3.lvd1689m",
                pretrained=True,
                img_size=224,
                device=device,
            ),
            layers=[
                "blocks.23",
            ],
        )
    )

    return GoldDescriptor(
        table_path=table_name,
        embedder=embedder,
        vectorizer=get_vit_class_token_vectorizer(),
        collate_fn=collate_cifar10,
        to_keep_schema=to_keep_schema,
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def get_gold_splitter(
    splitter_cfg: DictConfig,
    name_prefix: str,
    val_ratio: float,
    max_batches: int | None = None,
) -> GoldSplitter:
    splitter_config = hydra.utils.instantiate(splitter_cfg)

    batch_size = splitter_config["batch_size"]
    num_workers = splitter_config["num_workers"]
    min_pxt_insert_size = splitter_config["min_pxt_insert_size"]
    n_clusters = splitter_config["n_clusters"]

    to_keep_schema = {"label": pxt.String}

    table_name = f"{name_prefix}_{splitter_config["table_name"]}"

    clusterizer = (
        None
        if n_clusters < 2
        else GoldClusterizer(
            table_path=f"{table_name}_cluster",
            clustering_tool=GoldSKLearnClusteringTool(
                KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            ),
            vectorized_key="embeddings",
            min_pxt_insert_size=min_pxt_insert_size,
            batch_size=batch_size,
            num_workers=num_workers,
            to_keep_schema=to_keep_schema,
        )
    )

    descriptor = get_gold_descriptor(
        table_name=f"{table_name}_description",
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=batch_size,
        num_workers=num_workers,
        to_keep_schema=to_keep_schema,
    )

    selector = GoldSelector(
        table_path=f"{table_name}_selection",
        selection_tool=GoldGreedyKCenterSelectionTool(
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        reducer=None,
        vectorized_key="embeddings",
        label_key="label",
        to_keep_schema=to_keep_schema,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    sets = [
        GoldSet(name="train", size=1 - val_ratio),
        GoldSet(name="val", size=val_ratio),
    ]

    return GoldSplitter(
        sets=sets,
        descriptor=descriptor,
        vectorizer=None,
        clusterizer=clusterizer,
        n_clusters=n_clusters,
        selector=selector,
        max_batches=max_batches,
    )
