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
) -> dict:
    # make sure the transformer head is the same for all runs
    seed_everything(cfg.random_state)
    model = Cifar10LightningModule(
        learning_rate=cfg.learning_rate, model_type=cfg.model_type
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=f"{cfg.mlflow_experiment_name}_{data_module.settings_as_str}",
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=f"{cfg.mlflow_run_name}_{split_method}_{cfg.model_type}_{cfg.random_split_state}",
        log_model=True,
    )

    # Log additional parameters
    mlflow_logger.log_hyperparams(
        {
            "split_method": split_method,
            "val_ratio": cfg.val_ratio,
            "random_state": cfg.random_state,
            "remove_ratio": cfg.remove_ratio,
            "to_duplicate_clusters": cfg.to_duplicate_clusters,
            "cluster_count": cfg.cluster_count,
            "duplicate_per_sample": cfg.duplicate_per_sample,
            "random_split_state": cfg.random_split_state,
            "max_epochs": cfg.max_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "splitting_duration": splitting_duration,
            "splitting_update_selection": cfg.gold_splitter.update_selection,
            "model_type": cfg.model_type,
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

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        dirpath=f"./checkpoints/cifar10_{split_method}_split",
        filename="cifar10-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=1,
        mode="max",
        verbose=True,
        every_n_epochs=1,
    )

    # Initialize trainer
    debug_train_count = cfg.debug_train_count
    debug_count = cfg.debug_count
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
        log_every_n_steps=50,
        limit_train_batches=debug_train_count if debug_train_count is not None else 1.0,
        limit_val_batches=debug_count if debug_count is not None else 1.0,
        limit_test_batches=debug_count if debug_count is not None else 1.0,
    )

    # Train the model
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training with {split_method.upper()} split method")
    logger.info(f"{'=' * 60}\n")

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
    if cfg.validate_on_test:
        data_module.setup(stage="test")
        val_dataloaders["test_as_val"] = data_module.test_dataloader()

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=CombinedLoader(val_dataloaders, mode="max_size"),
    )

    if not cfg.validate_on_test:
        data_module.setup(stage="test")

    test_results = trainer.test(model, data_module, ckpt_path="best")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training completed for {split_method.upper()} split method")
    logger.info(f"Best validation AUROC: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"{'=' * 60}\n")

    best_val_auroc = checkpoint_callback.best_model_score
    assert best_val_auroc is not None
    return {
        "split_method": split_method,
        "best_val_auroc": best_val_auroc.item(),
        "test_results": test_results[0]["test_auroc"],
    }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run experiments with different split strategies
    Uses Hydra for configuration management
    """
    # Print configuration
    logger.info("Starting CIFAR-10 Split Comparison Experiment")
    logger.info(f"Configuration:\n{cfg}")

    # Create necessary directories
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    results = []

    seed_everything(cfg.random_state, workers=True)

    starts = time.monotonic()
    data_module = CIFAR10DataModule(
        cfg=cfg,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    splitting_duration = time.monotonic() - starts
    # Run experiments based on split method argument
    if cfg.split_method == "all":
        split_methods = [
            "gold",
            "random",
        ]
    else:
        split_methods = [cfg.split_method]

    for split_method in split_methods:
        result = run_experiment(
            split_method=split_method,
            data_module=data_module,
            cfg=cfg,
            splitting_duration=splitting_duration,
        )
        results.append(result)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for result in results:
        logger.info(f"\n{result['split_method'].upper()} Split Method:")
        logger.info(f"  Best Validation AUROC: {result['best_val_auroc']:.4f}")
        logger.info(f"  Test AUROC: {result['test_results']:.4f}")
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
