"""
Pascal VOC Segmentation Split Comparison Experiment

This script allows training different segmentation models on the Pascal VOC dataset using two different data splitting strategies:
1. Random split from scikit-learn
2. Smart split using GoldSplitter from the Goldener library

The key difference from image classification is that the splitting is done based on
patches corresponding to the segmentation mask (ground truth/target) rather than class tokens.
"""

import logging
import os
import time
from logging import getLogger
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader


from image_segmentation_pascal_voc.data import VOCSegmentationDataModule
from image_segmentation_pascal_voc.model import VOCSegmentationLightningModule


logger = getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)


def run_experiment(
    cfg: DictConfig,
    data_module: VOCSegmentationDataModule,
    splitting_duration: float,
    split_method: str = "random",
) -> None:
    model_type = cfg.exp.model
    logger.info(
        f"Running experiment with {split_method.upper()} split method and {model_type.upper()} model"
    )

    seed_everything(cfg.exp.random_state)
    model = VOCSegmentationLightningModule(
        learning_rate=cfg.exp.learning_rate, model_type=model_type
    )

    if cfg.exp.load_from_run_id:
        mlflow_client = mlflow.MlflowClient(
            tracking_uri=cfg.logging.mlflow_tracking_uri
        )
        run = mlflow_client.get_run(cfg.exp.load_from_run_id)
        artifact_path = Path(run.info.artifact_uri.replace("file://", ""))
        checkpoint_path = artifact_path / "voc-seg-best/voc-seg-best.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path} for run ID {cfg.exp.load_from_run_id}"
            )
        logger.info(
            f"Loading model from checkpoint at {checkpoint_path} for run ID {cfg.exp.load_from_run_id}"
        )
        model = VOCSegmentationLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            weights_only=False,
            hparams_file=artifact_path / "voc-seg-best/metadata.yaml",
            learning_rate=cfg.exp.learning_rate,
            model_type=model_type,
        )

    mlflow_logger = MLFlowLogger(
        experiment_name=f"{cfg.logging.mlflow_experiment_name}_{data_module.settings_as_str}",
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        run_name=f"{cfg.logging.mlflow_run_name}_{split_method}_{model_type}_{cfg.data.random_split_state}_{cfg.data.random_shuffle_state}",
        log_model=True,
    )

    mlflow_logger.log_hyperparams(
        {
            "split_method": split_method,
            "val_ratio": cfg.exp.val_ratio,
            "random_state": cfg.exp.random_state,
            "remove_ratio": cfg.data.remove_ratio,
            "to_duplicate_clusters": cfg.data.to_duplicate_clusters,
            "cluster_count": cfg.data.cluster_count,
            "duplicate_per_sample": cfg.data.duplicate_per_sample,
            "random_split_state": cfg.data.random_split_state,
            "random_shuffle_state": cfg.data.random_shuffle_state,
            "max_epochs": cfg.exp.max_epochs,
            "batch_size": cfg.exp.batch_size,
            "learning_rate": cfg.exp.learning_rate,
            "splitting_duration": splitting_duration,
            "splitting_update_selection": cfg.gold_splitter.update_selection,
            "model_type": model_type,
            "n_clusters": cfg.gold_splitter.n_clusters,
        }
    )

    mlflow_logger.experiment.log_dict(
        run_id=mlflow_logger.run_id,
        dictionary=dict(cfg),
        artifact_file="config.yaml",
    )

    if split_method == "gold":
        indices_dict = {
            "gold_train_indices": data_module.gold_train_indices,
            "gold_val_indices": data_module.gold_val_indices,
        }
    else:
        indices_dict = {
            "sk_train_indices": data_module.sk_train_indices,
            "sk_val_indices": data_module.sk_val_indices,
        }

    mlflow_logger.experiment.log_dict(
        run_id=mlflow_logger.run_id,
        dictionary=indices_dict
        | {
            "duplicated": data_module.duplicated_train_indices,
            "excluded_indices": data_module.excluded_train_indices,
        },
        artifact_file="indices.json",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        dirpath=f"{cfg.exp.checkpoint}/{mlflow_logger.run_id}/",
        filename="voc-seg-best",
        save_top_k=1,
        mode="max",
        verbose=True,
        every_n_epochs=1,
    )

    debug_train_count = cfg.debug_count.train_count
    debug_test_count = cfg.debug_count.test_count
    trainer = Trainer(
        max_epochs=cfg.exp.max_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        deterministic=False,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        limit_train_batches=debug_train_count if debug_train_count is not None else 1.0,
        limit_val_batches=debug_test_count if debug_test_count is not None else 1.0,
        limit_test_batches=debug_test_count if debug_test_count is not None else 1.0,
    )

    train_dataloader = (
        data_module.sk_train_dataloader()
        if split_method == "random"
        else data_module.gold_train_dataloader()
    )
    val_dataloaders = {
        "val": (
            data_module.sk_val_dataloader()
            if split_method == "random"
            else data_module.gold_val_dataloader()
        )
    }
    if cfg.exp.validate_on_test:
        data_module.setup(stage="test")
        val_dataloaders["test_as_val"] = data_module.test_dataloader()

    if cfg.exp.load_from_run_id is None:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=CombinedLoader(val_dataloaders, mode="max_size"),
        )

    if not cfg.exp.validate_on_test or cfg.exp.load_from_run_id is not None:
        data_module.setup(stage="test")

    trainer.test(
        model,
        data_module,
        ckpt_path="best" if cfg.exp.load_from_run_id is None else None,
    )

    logger.info(
        f"Run completed for {split_method.upper()} split method and {model_type.upper()} model"
    )


@hydra.main(
    version_base="1.3.2",
    config_path="config",
    config_name="config",
)
def main(cfg: DictConfig):
    # Print configuration
    logger.info("Starting Pascal VOC Segmentation Split Comparison Experiment")
    logger.info(f"Configuration:\n{cfg}")

    # Create necessary directories
    os.makedirs(cfg.data.cache, exist_ok=True)
    os.makedirs(cfg.exp.checkpoint, exist_ok=True)

    seed_everything(cfg.exp.random_state, workers=True)

    # run data preparation and splitting to ensure that the same
    # splits are used for all experiments and to log the splitting duration
    starts = time.monotonic()
    data_module = VOCSegmentationDataModule(
        cfg=cfg,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")
    splitting_duration = time.monotonic() - starts
    logger.info(
        f"Data preparation and splitting completed in {splitting_duration:.2f} seconds"
    )

    # Run experiments based on split method argument
    if cfg.exp.split_method == "all":
        split_methods = [
            "gold",
            "random",
        ]
    else:
        split_methods = [cfg.exp.split_method]

    for split_method in split_methods:
        run_experiment(
            split_method=split_method,
            data_module=data_module,
            cfg=cfg,
            splitting_duration=splitting_duration,
        )


if __name__ == "__main__":
    main()
