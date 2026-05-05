import os
import time
from logging import getLogger, WARNING
from typing import Literal

import hydra
from omegaconf import DictConfig
import pixeltable as pxt
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from text_classification_imdb.data import IMDbDataModule
from text_classification_imdb.model import IMDbLightningModule


logger = getLogger(__name__)

pxt.configure_logging(to_stdout=True, level=WARNING, remove="goldener")


def run_experiment(
    cfg: DictConfig,
    data_module: IMDbDataModule,
    splitting_duration: float,
    split_method: Literal["random", "gold"] = "random",
    batch_method: Literal["random", "gold"] = "random",
) -> None:
    model_type = cfg.exp.model
    logger.info(
        f"Running experiment with {split_method.upper()} split method and {model_type.upper()} model"
    )

    seed_everything(cfg.exp.random_state)
    model = IMDbLightningModule(
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
            "batch_method": batch_method,
            "model_type": model_type,
            "val_ratio": cfg.exp.val_ratio,
            "max_epochs": cfg.exp.max_epochs,
            "learning_rate": cfg.exp.learning_rate,
            "batch_size": cfg.exp.batch_size,
            "random_state": cfg.exp.random_state,
            "tokenizer_name": cfg.data.tokenizer_name,
            "max_length": cfg.data.max_length,
            "random_split_state": cfg.data.random_split_state,
            "random_shuffle_state": cfg.data.random_shuffle_state,
            "splitting_duration": splitting_duration,
            "splitting_update_selection": cfg.goldener_config.update_selection,
            "n_clusters": cfg.goldener_config.n_clusters,
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
            "excluded_indices": data_module.excluded_train_indices,
        },
        artifact_file="indices.json",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        dirpath=f"{cfg.exp.checkpoint}/{mlflow_logger.run_id}/",
        filename="imdb-{epoch:02d}-{val_auroc:.4f}",
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
        data_module.sk_train_dataloader(batch_method=batch_method)
        if split_method == "random"
        else data_module.gold_train_dataloader(batch_method=batch_method)
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


ExperimentMethod = Literal["random", "gold"]


@hydra.main(
    version_base="1.3.2",
    config_path="config",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    logger.info("Starting IMDb Split Comparison Experiment")
    logger.info(f"Configuration:\n{cfg}")

    os.makedirs(cfg.exp.checkpoint, exist_ok=True)

    seed_everything(cfg.exp.random_state, workers=True)

    start_time = time.monotonic()
    data_module = IMDbDataModule(cfg=cfg)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    splitting_duration = time.monotonic() - start_time
    logger.info(
        f"Data preparation and splitting completed in {splitting_duration:.2f} seconds"
    )

    if cfg.exp.split_method == "all":
        split_methods = [
            "gold",
            "random",
        ]
    else:
        assert cfg.exp.split_method in ("random", "gold")
        split_methods = [cfg.exp.split_method]

    if cfg.exp.batch_method == "all":
        batch_methods = ["gold", "random"]
    else:
        assert cfg.exp.batch_method in ("random", "gold")
        batch_methods = [cfg.exp.batch_method]

    for split_method in split_methods:
        for batch_method in batch_methods:
            run_experiment(
                split_method=split_method,  # type: ignore[arg-type]
                batch_method=batch_method,  # type: ignore[arg-type]
                data_module=data_module,
                cfg=cfg,
                splitting_duration=splitting_duration,
            )


if __name__ == "__main__":
    main()
