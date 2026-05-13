import pixeltable as pxt
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Subset
from sklearn.cluster import KMeans
from transformers import AutoModel

from goldener import (
    GoldSKLearnClusteringTool,
    GoldClusterizer,
    GoldDescriptor,
    GoldTorchEmbeddingTool,
    GoldTorchEmbeddingToolConfig,
    GoldSelector,
    GoldGreedyKCenterSelectionTool,
    GoldSet,
    GoldSplitter,
    TensorVectorizer,
    Filter2DWithCount,
    FilterLocation,
    GoldSKLearnReductionTool,
)
from goldener.organize import GoldClusterizedBatchSampler, ExhaustedClusterStrategy

from omegaconf import DictConfig


def collate_imdb(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, int]],
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    input_ids, _, labels, indices = zip(*batch)
    ids_tensor = torch.stack(list(input_ids))
    str_labels = [str(label) for label in labels]
    idx_list = [int(idx) for idx in indices]
    return {
        "data": ids_tensor,
        "label": str_labels,
        "idx": idx_list,
    }


def get_gold_descriptor(
    table_name: str,
    min_pxt_insert_size: int,
    batch_size: int,
    num_workers: int,
    to_keep_schema: dict | None = None,
    pretrained_model: str = "google-bert/bert-base-uncased",
    max_batches: int | None = None,
) -> GoldDescriptor:
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    model = AutoModel.from_pretrained(pretrained_model).to(device)
    embedder = GoldTorchEmbeddingTool(
        GoldTorchEmbeddingToolConfig(
            model=model,
            layers=["encoder.layer.11"],
        )
    )

    tensor_vectorizer = TensorVectorizer(
        keep=Filter2DWithCount(filter_location=FilterLocation.START, keep=True),
        channel_pos=2,
    )

    return GoldDescriptor(
        table_path=table_name,
        embedder=embedder,
        vectorizer=tensor_vectorizer,
        collate_fn=collate_imdb,
        to_keep_schema=to_keep_schema,
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_batches=max_batches,
    )


def get_gold_splitter(
    goldener_config: DictConfig,
    name_prefix: str,
    val_ratio: float,
    max_batches: int | None = None,
) -> GoldSplitter:
    batch_size = goldener_config.batch_size
    num_workers = goldener_config.num_workers
    min_pxt_insert_size = goldener_config.min_pxt_insert_size
    n_clusters = goldener_config.n_clusters
    pretrained_model = goldener_config.pretrained_model

    to_keep_schema = {"label": pxt.String}

    table_name = f"{name_prefix}_{goldener_config.table_name}"

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
        pretrained_model=pretrained_model,
        max_batches=max_batches,
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
        n_clusters=n_clusters,
        clusterizer=clusterizer,
        selector=selector,
        max_batches=max_batches,
    )


def get_gold_batcher(
    dataset: Subset,
    goldener_config: DictConfig,
    name_prefix: str,
    batch_size: int,
    generator: torch.Generator,
    max_batches: int | None = None,
    update_batch: bool = True,
) -> GoldClusterizedBatchSampler:
    goldener_batch_size = goldener_config.batch_size
    num_workers = goldener_config.num_workers
    min_pxt_insert_size = goldener_config.min_pxt_insert_size
    pretrained_model = goldener_config.pretrained_model
    n_clusters = goldener_config.n_clusters_batcher

    table_name = f"{name_prefix}_{goldener_config.table_name}"
    cluster_table_path = f"{table_name}_batcher_cluster"
    description_table_path = f"{table_name}_batcher_description"
    if update_batch:
        pxt.drop_table(cluster_table_path, if_not_exists="ignore")
        pxt.drop_table(description_table_path, if_not_exists="ignore")

    sklearn_tool = KMeans(
        n_clusters=batch_size,
        random_state=42,
    )
    reducer = GoldSKLearnReductionTool(PCA(n_components=2, random_state=0))

    clusterizer = GoldClusterizer(
        table_path=cluster_table_path,
        clustering_tool=GoldSKLearnClusteringTool(
            tool=sklearn_tool,
        ),
        reducer=reducer,
        vectorized_key="embeddings",
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=goldener_batch_size,
        num_workers=num_workers,
    )

    descriptor = get_gold_descriptor(
        table_name=description_table_path,
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=goldener_batch_size,
        num_workers=num_workers,
        pretrained_model=pretrained_model,
        max_batches=max_batches,
    )

    return GoldClusterizedBatchSampler(
        dataset=dataset,
        descriptor=descriptor,
        vectorizer=None,
        batch_size=batch_size,
        n_clusters=n_clusters,
        clusterizer=clusterizer,
        force_same_size=False,
        shuffle=True,
        generator=generator,
        strategy=ExhaustedClusterStrategy.EXCLUDE,
    )
