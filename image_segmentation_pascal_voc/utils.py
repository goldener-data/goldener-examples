import pixeltable as pxt

import hydra
import timm
import torch
from sklearn.cluster import KMeans
from any_gold.tools.image.utils import gold_multi_class_segmentation_collate_fn
from goldener.vision.vectorizers import get_vit_patch_tokens_vectorizer
from goldener import (
    GoldSKLearnClusteringTool,
    GoldClusterizer,
    GoldDescriptor,
    TorchGoldFeatureExtractor,
    TorchGoldFeatureExtractorConfig,
    GoldSelector,
    GoldGreedyKCenterSelection,
    GoldSet,
    GoldSplitter,
)
from omegaconf import DictConfig
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, Resize

PASCAL_VOC_PREPROCESS = Compose(
    [
        ToTensor(),
        Resize(224),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def get_gold_descriptor(
    table_name: str,
    min_pxt_insert_size: int,
    batch_size: int,
    num_workers: int,
    to_keep_schema: dict,
    target_to_label: dict[tuple[int, int, int], str],
) -> GoldDescriptor:
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )

    extractor = TorchGoldFeatureExtractor(
        TorchGoldFeatureExtractorConfig(
            model=timm.create_model(
                model_name="vit_small_patch16_dinov3.lvd1689m",
                pretrained=True,
                img_size=224,
                device=device,
            ),
            layers=["blocks.11"],
        )
    )

    # For segmentation, we want to extract features from all patches, not just the class token
    # We'll use ALL filter location to get all patch embeddings (excluding class token)
    # Then we can filter based on the segmentation mask
    return GoldDescriptor(
        table_path=table_name,
        extractor=extractor,
        vectorizer=get_vit_patch_tokens_vectorizer(),
        collate_fn=gold_multi_class_segmentation_collate_fn,
        to_keep_schema=to_keep_schema,
        target_to_label=target_to_label,
        label_key="label",
        exclude_full_zero_target=True,
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def get_gold_splitter(
    splitter_cfg: DictConfig,
    name_prefix: str,
    val_ratio: float,
    target_to_label: dict[tuple[int, int, int], str],
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
            vectorized_key="features",
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
        target_to_label=target_to_label,
    )

    selector = GoldSelector(
        table_path=f"{table_name}_selection",
        selection_tool=GoldGreedyKCenterSelection(
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        reducer=None,
        vectorized_key="features",
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
        selector=selector,
        max_batches=max_batches,
    )
