"""
CIFAR-10 Split Comparison Experiment

This script allow to train different models on the CIFAR-10 dataset using two different data splitting strategies:
1. Random split from scikit-learn
2. Smart split using GoldSplitter from the Goldener library

"""

import os
import time
from logging import getLogger, WARNING

import hydra
from omegaconf import DictConfig
import pixeltable as pxt
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader


from image_classification_cifar10.data import CIFAR10DataModule
from image_classification_cifar10.model import Cifar10LightningModule


logger = getLogger(__name__)

pxt.configure_logging(to_stdout=True, level=WARNING, remove="goldener")


def run_experiment(
    cfg: DictConfig,
    data_module: CIFAR10DataModule,
    splitting_duration: float,
    split_method: str = "random",
) -> None:
    model_type = cfg.exp.model
    logger.info(
        f"Running experiment with {split_method.upper()} split method and {model_type.upper()} model"
    )

    seed_everything(cfg.exp.random_state)
    model = Cifar10LightningModule(
        learning_rate=cfg.exp.learning_rate, model_type=model_type
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=f"{cfg.logging.mlflow_experiment_name}_{data_module.settings_as_str}",
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        run_name=f"{cfg.logging.mlflow_run_name}_{split_method}_{model_type}_{cfg.data.random_split_state}",
        log_model=True,
    )

    mlflow_logger.log_hyperparams(
        {
            "split_method": split_method,
            "model_type": model_type,
            "val_ratio": cfg.exp.val_ratio,
            "max_epochs": cfg.exp.max_epochs,
            "learning_rate": cfg.exp.learning_rate,
            "batch_size": cfg.exp.batch_size,
            "random_state": cfg.exp.random_state,
            "remove_ratio": cfg.data.remove_ratio,
            "drop_duplicate_table": cfg.data.drop_duplicate_table,
            "to_duplicate_clusters": cfg.data.to_duplicate_clusters,
            "cluster_count": cfg.data.cluster_count,
            "duplicate_per_sample": cfg.data.duplicate_per_sample,
            "random_split_state": cfg.data.random_split_state,
            "splitting_duration": splitting_duration,
            "splitting_update_selection": cfg.gold_splitter.update_selection,
            "n_clusters": cfg.gold_splitter.n_clusters,
        }
    )

    mlflow_logger.experiment.log_dict(
        run_id=mlflow_logger.run_id,
        dictionary=dict(cfg),
        artifact_file="config.yaml",
    )

    mlflow_logger.experiment.log_dict(
        run_id=mlflow_logger.run_id,
        dictionary={
            "gold_train_indices": data_module.gold_train_indices,
            "gold_val_indices": data_module.gold_val_indices,
            "sk_train_indices": data_module.sk_train_indices,
            "sk_val_indices": data_module.sk_val_indices,
            "duplicated": data_module.duplicated_train_indices,
            "excluded_indices": data_module.excluded_train_indices,
        },
        artifact_file="indices.json",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        dirpath=f"{cfg.exp.checkpoint}/{mlflow_logger.run_id}/",
        filename="cifar10-{epoch:02d}-{val_auroc:.4f}",
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
        deterministic=True,
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

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=CombinedLoader(val_dataloaders, mode="max_size"),
    )

    if not cfg.exp.validate_on_test:
        data_module.setup(stage="test")

    trainer.test(model, data_module, ckpt_path="best")

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
    logger.info("Starting CIFAR-10 Split Comparison Experiment")
    logger.info(f"Configuration:\n{cfg}")

    # Create necessary directories
    os.makedirs(cfg.data.cache, exist_ok=True)
    os.makedirs(cfg.exp.checkpoint, exist_ok=True)

    seed_everything(cfg.exp.random_state, workers=True)

    # run data preparation and splitting to ensure that the same
    # splits are used for all experiments and to log the splitting duration
    starts = time.monotonic()
    data_module = CIFAR10DataModule(
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
