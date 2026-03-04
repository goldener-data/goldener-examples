import pixeltable as pxt

import hydra
import timm
import torch
from goldener.torch_utils import get_unique_values_in_tensor
from sklearn.cluster import KMeans
from goldener.vision.transform import PatchifyImageMask
from goldener.vision.vectorizers import get_vit_patch_tokens_vectorizer
from goldener import (
    GoldSKLearnClusteringTool,
    GoldClusterizer,
    GoldDescriptor,
    TorchGoldEmbeddingTool,
    TorchGoldEmbeddingToolConfig,
    GoldSelector,
    GoldGreedyKCenterSelectionTool,
    GoldSet,
    GoldSplitter,
)
from omegaconf import DictConfig
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

PASCAL_VOC_PREPROCESS = Compose(
    [
        ToImage(),
        ToDtype(torch.float32, scale=False),
        Resize(
            (224, 224),
        ),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def collate_pascal_voc(
    batch: list[dict[str, torch.Tensor | set[str] | int]],
) -> dict[str, torch.Tensor | list[list[str]] | list[int]]:
    images = []
    masks = []
    labels: list[list[str]] = []
    indices = []
    for item in batch:
        image = item["image"]
        assert isinstance(image, torch.Tensor)
        images.append(image)

        mask = item["mask"]
        assert isinstance(mask, torch.Tensor)
        masks.append(mask)

        labels_input = item["labels"]
        assert isinstance(labels_input, set)
        labels.append(list(labels_input))

        index = item["index"]
        assert isinstance(index, int)
        indices.append(index)

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "labels": labels,
        "idx": indices,
    }


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

    embedder = TorchGoldEmbeddingTool(
        TorchGoldEmbeddingToolConfig(
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
    patchify_mask = PatchifyImageMask(patch_size=16, match_ratio=0.85)

    return GoldDescriptor(
        table_path=table_name,
        embedder=embedder,
        vectorizer=get_vit_patch_tokens_vectorizer(
            transform_y=patchify_mask.transform,
            n_random=5,
            generator=torch.Generator().manual_seed(42),
        ),
        collate_fn=collate_pascal_voc,
        to_keep_schema=to_keep_schema,
        data_key="image",
        target_key="mask",
        target_to_label=target_to_label,
        label_key="labels",
        exclude_full_zero_target=True,
        exclude_labels={"void", "background"},
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

    to_keep_schema = {"labels": pxt.String}

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
        selection_tool=GoldGreedyKCenterSelectionTool(
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        reducer=None,
        vectorized_key="features",
        label_key="labels",
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


def transform_rgb_mask_to_class_mask(
    y: torch.Tensor, rgb_to_class_idx: dict[tuple[int, int, int], int]
) -> torch.Tensor:
    b, _, h, w = y.shape
    new_y = torch.zeros((b, 21, h, w), dtype=y.dtype, device=y.device)
    unique_targets = get_unique_values_in_tensor(y)

    for unique_target in unique_targets:
        unique_target_tuple = tuple(unique_target.tolist())
        if unique_target_tuple not in rgb_to_class_idx:
            raise ValueError(
                f"Unique target {unique_target_tuple} not found in target_to_label mapping."
            )
        r, g, b = unique_target_tuple
        mask = (y[:, 0] == r) & (y[:, 1] == g) & (y[:, 2] == b)
        new_index = rgb_to_class_idx[unique_target_tuple]
        new_y[:, new_index][mask] = 1

    return new_y.float()


def transform_segmentation_logits_to_rgb_preds(
    logits: torch.Tensor,
    rgb_to_class_idx: dict[tuple[int, int, int], int],
) -> torch.Tensor:
    b, _, h, w = logits.shape
    preds = torch.argmax(logits, dim=1)
    rgb_preds = torch.zeros((b, 3, h, w), dtype=torch.long, device=logits.device)

    for rgb_tuple, class_index in rgb_to_class_idx.items():
        mask = preds == class_index
        r, g, b = rgb_tuple
        rgb_preds[:, 0][mask] = r
        rgb_preds[:, 1][mask] = g
        rgb_preds[:, 2][mask] = b

    return rgb_preds


def transform_rgb_mask_to_mono_mask(
    rgb_mask: torch.Tensor,
    rgb_to_class_idx: dict[tuple[int, int, int], int],
) -> torch.Tensor:
    b, _, h, w = rgb_mask.shape
    mono_mask = torch.zeros((b, 1, h, w), dtype=torch.long, device=rgb_mask.device)

    for rgb_tuple, class_index in rgb_to_class_idx.items():
        r, g, b = rgb_tuple
        mask = (rgb_mask[:, 0] == r) & (rgb_mask[:, 1] == g) & (rgb_mask[:, 2] == b)
        mono_mask[mask.unsqueeze(1)] = class_index

    return mono_mask
