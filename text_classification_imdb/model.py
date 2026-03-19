from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from transformers import AutoModel


logger = getLogger(__name__)


class TextCNNModel(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        kernel_sizes = [3, 4, 5]
        self.embedding = nn.Embedding(30522, 128, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(128, 128, kernel_size=k),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * len(kernel_sizes), 1)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        pooled = [F.adaptive_max_pool1d(conv(x), 1).squeeze(-1) for conv in self.convs]
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)  # (batch,)


class BertClassifier(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.classifier(x).squeeze(-1)  # (batch,)


class IMDbLightningModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 2e-5,
        model_type: str = "cnn",
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.model: torch.nn.Module
        self._setup_model(model_type)

        self.save_hyperparameters()

    def _setup_model(self, model_type: str) -> None:
        if model_type == "cnn":
            self.model = TextCNNModel()
        elif model_type == "bert":
            self.model = BertClassifier()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids)

    def on_train_start(self) -> None:
        self.train_auroc = BinaryAUROC()
        self.train_acc = BinaryAccuracy().to(
            torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        )

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        auroc_metric: BinaryAUROC,
        acc_metric: BinaryAccuracy,
        prefix: str,
    ) -> torch.Tensor:
        input_ids, attention_mask, labels, _ = batch

        logits = self(input_ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(logits)
        auroc_metric.update(probs, labels)
        acc_metric.update(probs, labels)

        return loss

    def _compute_metrics_and_log(
        self,
        auroc_metric: BinaryAUROC,
        acc_metric: BinaryAccuracy,
        prefix: str,
    ) -> None:
        auroc = auroc_metric.compute()
        acc = acc_metric.compute()
        self.log(f"{prefix}_auroc", auroc, prog_bar=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True)
        auroc_metric.reset()
        acc_metric.reset()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        return self._step(
            batch=batch,
            auroc_metric=self.train_auroc,
            acc_metric=self.train_acc,
            prefix="train",
        )

    def on_train_epoch_end(self) -> None:
        self._compute_metrics_and_log(self.train_auroc, self.train_acc, "train")

    def on_validation_epoch_start(self) -> None:
        self.val_auroc = BinaryAUROC()
        self.val_acc = BinaryAccuracy().to(
            torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        )
        if self.has_test_as_val:
            self.test_as_val_auroc = BinaryAUROC()
            self.test_as_val_acc = BinaryAccuracy().to(
                torch.device("cpu" if not torch.cuda.is_available() else "cuda")
            )

    @property
    def has_test_as_val(self) -> bool:
        if not isinstance(self.trainer.val_dataloaders, dict):
            return False
        return "test_as_val" in self.trainer.val_dataloaders

    def validation_step(
        self,
        batch: (
            dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ),
        batch_idx: int,
    ) -> STEP_OUTPUT:
        if isinstance(batch, dict):
            val_batch = batch["val"]
            test_batch = batch.get("test_as_val")
        else:
            val_batch = batch
            test_batch = None

        loss = None
        if val_batch is not None:
            loss = self._step(
                batch=val_batch,
                auroc_metric=self.val_auroc,
                acc_metric=self.val_acc,
                prefix="val",
            )

        if test_batch is not None:
            self._step(
                batch=test_batch,
                auroc_metric=self.test_as_val_auroc,
                acc_metric=self.test_as_val_acc,
                prefix="test_as_val",
            )

        return loss

    def on_validation_epoch_end(self) -> None:
        self._compute_metrics_and_log(self.val_auroc, self.val_acc, "val")
        if self.has_test_as_val:
            self._compute_metrics_and_log(
                self.test_as_val_auroc, self.test_as_val_acc, "test_as_val"
            )

    def on_test_start(self) -> None:
        self.test_auroc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        return self._step(
            batch=batch,
            auroc_metric=self.test_auroc,
            acc_metric=self.test_acc,
            prefix="test",
        )

    def on_test_epoch_end(self) -> None:
        self._compute_metrics_and_log(self.test_auroc, self.test_acc, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
