from lightning import LightningModule
import torch
import segmentation_models_pytorch as smp
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

from image_segmentation_pascal_voc.utils import (
    transform_rgb_mask_to_class_mask,
    transform_segmentation_logits_to_rgb_preds,
    transform_rgb_mask_to_mono_mask,
)


class VOCSegmentationLightningModule(LightningModule):
    RGB_TO_CLASS_IDX = {
        (0, 0, 0): 0,  # background
        (128, 0, 0): 1,
        (0, 128, 0): 2,
        (128, 128, 0): 3,
        (0, 0, 128): 4,
        (128, 0, 128): 5,
        (0, 128, 128): 6,
        (128, 128, 128): 7,
        (64, 0, 0): 8,
        (192, 0, 0): 9,
        (64, 128, 0): 10,
        (192, 128, 0): 11,
        (64, 0, 128): 12,
        (192, 0, 128): 13,
        (64, 128, 128): 14,
        (192, 128, 128): 15,
        (0, 64, 0): 16,
        (128, 64, 0): 17,
        (0, 192, 0): 18,
        (128, 192, 0): 19,
        (0, 64, 128): 20,
        (224, 224, 192): 0,  # void
    }

    def __init__(
        self,
        learning_rate: float = 0.001,
        model_type: str = "unet",
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate

        self.model: torch.nn.Module
        self._setup_model(model_type)

        self.save_hyperparameters()

    def _setup_model(self, model_type: str) -> None:
        if model_type == "deeplab":
            self.model = smp.DeepLabV3Plus(
                encoder_name="mobilenet_v2",
                in_channels=3,
                classes=21,
            )
        elif model_type == "fpn":
            self.model = smp.FPN(
                encoder_name="mobilenet_v2",
                classes=21,
                in_channels=3,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model_type = model_type

    @property
    def has_test_as_val(self) -> bool:
        if not isinstance(self.trainer.val_dataloaders, dict):
            return False
        return "test_as_val" in self.trainer.val_dataloaders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self.train_iou = MulticlassJaccardIndex(num_classes=21).to(self.device)
        self.train_micro_iou = MulticlassJaccardIndex(
            num_classes=21, average="micro"
        ).to(self.device)
        self.train_pc_iou = MulticlassJaccardIndex(num_classes=21, average=None).to(
            self.device
        )

    def _step(
        self,
        batch: dict[str, torch.Tensor | list[list[str]] | list[int]],
        iou_metric: MulticlassJaccardIndex,
        iou_pc_metric: MulticlassJaccardIndex,
        iou_micro_metric: MulticlassJaccardIndex,
        prefix: str,
    ) -> torch.Tensor:
        x = batch["image"]
        assert isinstance(x, torch.Tensor)
        y = batch["mask"]
        assert isinstance(y, torch.Tensor)

        logits = self(x)

        seg_y = transform_rgb_mask_to_class_mask(y, self.RGB_TO_CLASS_IDX)

        loss = F.cross_entropy(logits, seg_y)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = transform_segmentation_logits_to_rgb_preds(
            logits, self.RGB_TO_CLASS_IDX
        )
        preds_mono = transform_rgb_mask_to_mono_mask(preds, self.RGB_TO_CLASS_IDX)
        y_mono = transform_rgb_mask_to_mono_mask(y, self.RGB_TO_CLASS_IDX)

        acc = (preds_mono == y_mono).float().mean()
        self.log(f"{prefix}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        iou_metric.update(
            preds_mono,
            y_mono,
        )
        iou_pc_metric.update(
            preds_mono,
            y_mono,
        )
        iou_micro_metric.update(
            preds_mono,
            y_mono,
        )

        return loss

    def _compute_iou_and_log(
        self,
        iou_metric: MulticlassJaccardIndex,
        prefix: str,
    ) -> None:
        iou = iou_metric.compute()

        if iou.ndim == 0:
            self.log(f"{prefix}_iou", iou, prog_bar=True)
        else:
            for i, class_iou in enumerate(iou):
                self.log(f"{prefix}_iou_class_{i}", class_iou, prog_bar=False)

        iou_metric.reset()

    def training_step(
        self,
        batch: dict[str, torch.Tensor | list[list[str]] | list[int]],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        return self._step(
            batch=batch,
            iou_metric=self.train_iou,
            iou_pc_metric=self.train_pc_iou,
            iou_micro_metric=self.train_micro_iou,
            prefix="train",
        )

    def on_train_epoch_end(self) -> None:
        self._compute_iou_and_log(self.train_iou, "train")
        self._compute_iou_and_log(self.train_pc_iou, "train")
        self._compute_iou_and_log(self.train_micro_iou, "train_micro")

    def on_validation_epoch_start(self) -> None:
        self.val_iou = MulticlassJaccardIndex(num_classes=21).to(self.device)
        self.val_pc_iou = MulticlassJaccardIndex(num_classes=21, average=None).to(
            self.device
        )
        self.val_micro_iou = MulticlassJaccardIndex(num_classes=21, average="micro").to(
            self.device
        )

        if self.has_test_as_val:
            self.test_as_val_iou = MulticlassJaccardIndex(num_classes=21).to(
                self.device
            )
            self.test_as_val_pc_iou = MulticlassJaccardIndex(
                num_classes=21, average=None
            ).to(self.device)
            self.test_as_val_micro_iou = MulticlassJaccardIndex(
                num_classes=21, average="micro"
            ).to(self.device)

    def validation_step(
        self,
        batch: dict[str, dict[str, torch.Tensor | list[list[str]] | list[int]]]
        | dict[str, torch.Tensor | list[list[str]] | list[int]],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        if "val" in batch:
            val_batch = batch["val"]
            test_batch = batch["test_as_val"] if "test_as_val" in batch else None
        else:
            val_batch = batch  # type: ignore[assignment]
            test_batch = None

        loss = None
        if val_batch is not None:
            loss = self._step(
                batch=val_batch,  # type: ignore[arg-type]
                iou_metric=self.val_iou,
                iou_pc_metric=self.val_pc_iou,
                iou_micro_metric=self.val_micro_iou,
                prefix="val",
            )

        if test_batch is not None:
            self._step(
                batch=test_batch,  # type: ignore[arg-type]
                iou_metric=self.test_as_val_iou,
                iou_pc_metric=self.test_as_val_pc_iou,
                iou_micro_metric=self.test_as_val_micro_iou,
                prefix="test_as_val",
            )

        return loss

    def on_validation_epoch_end(self) -> None:
        self._compute_iou_and_log(self.val_iou, "val")
        self._compute_iou_and_log(self.val_pc_iou, "val")
        self._compute_iou_and_log(self.val_micro_iou, "val_micro")

        if self.has_test_as_val:
            self._compute_iou_and_log(self.test_as_val_iou, "test_as_val")
            self._compute_iou_and_log(self.test_as_val_pc_iou, "test_as_val")
            self._compute_iou_and_log(self.test_as_val_micro_iou, "test_as_val_micro")

    def on_test_start(self) -> None:
        self.test_iou = MulticlassJaccardIndex(num_classes=21).to("cuda")
        self.test_pc_iou = MulticlassJaccardIndex(num_classes=21, average=None).to(
            "cuda"
        )
        self.test_micro_iou = MulticlassJaccardIndex(
            num_classes=21, average="micro"
        ).to("cuda")

    def test_step(
        self,
        batch: dict[str, torch.Tensor | list[list[str]] | list[int]],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        return self._step(
            batch=batch,
            iou_metric=self.test_iou,
            iou_pc_metric=self.test_pc_iou,
            iou_micro_metric=self.test_micro_iou,
            prefix="test",
        )

    def on_test_epoch_end(self) -> None:
        self._compute_iou_and_log(
            iou_metric=self.test_iou,
            prefix="test",
        )
        self._compute_iou_and_log(
            iou_metric=self.test_pc_iou,
            prefix="test_pc",
        )
        self._compute_iou_and_log(
            iou_metric=self.test_micro_iou,
            prefix="test_micro",
        )

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer
