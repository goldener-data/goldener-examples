# Pascal VOC Segmentation Split Comparison Experiment

This experiment compares two data splitting strategies for training
image segmentation models on the [Pascal VOC 2012 dataset](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/index.html).

## Table of Contents

- [Main Components](#main-components)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [Split Strategies](#split-strategies)
- [Viewing Results](#viewing-results)


## Main Components

- **Configuration**: The experiment is configured from a config file loaded from Hydra for flexible configuration management.
It allows specifying the hyperparameters and logging parameters for the model training/evaluation
as well as the data split method to use and the settings for the GoldSplitter.

- **VOCSegmentationDataModule**: A specific Pytorch Lightning DataModule allowing to load data from the Pascal VOC 2012 dataset
(1,464 training images, 1,449 validation images). Depending on the configuration, only a subset of the
training images is used for training/validation. Duplication of some samples is as well possible.

- **VOCSegmentationLightningModule**: A specific Pytorch Lightning LightningModule allowing to train and evaluate different
image segmentation models (deeplab and fpn segmentation models) for the Pascal VOC dataset.

- **Trainer**: PyTorch Lightning Trainer for efficient training management allowing to handle training, validation and
testing loops. It checkpoints the best model based on validation IoU metric.

- **Logging**: MLFlow for experiment tracking allowing to compare the different splitting strategies based on the logged metrics.

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Make sure you're in the experiment directory
cd image_segmentation_pascal_voc

# Run both split methods (uses default config)
uv run python voc_experiment.py

# View results
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.


## Technical Details

### Dataset: Pascal VOC 2012

- **Task**: Semantic segmentation
- **Classes**: 21 classes (20 object categories + background)
  - Background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
    dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, tv/monitor
- **Training samples**: 1,464 images with segmentation masks
- **Validation samples**: 1,449 images with segmentation masks
- **Image size**: Variable, resized to 224×224 for training


### Training Configuration

- **Optimizer**: Adam
  - Learning rate: 0.0001 (default, configurable)
  - Weight decay: 0 (no L2 regularization)

- **Loss Function**: CrossEntropyLoss
  - Applied to pixel-wise predictions

- **Batch Size**: 16 (default, configurable)
- **Max Epochs**: 60 (default, configurable)

### Evaluation Metrics

**Primary Metric (for model selection)**:
- **IoU (Intersection over Union)**: Also known as Jaccard Index, measures the overlap between predicted and ground truth segmentation masks
  - Task: Multiclass segmentation
  - Used for: Model checkpoint selection (best validation IoU)
  - Formula: IoU = |A ∩ B| / |A ∪ B|

**Secondary Metrics**:
- **Pixel Accuracy**: Percentage of correctly classified pixels

All metrics are computed and logged for both training and validation sets at each epoch.

### Model Selection

The best model is selected based on **maximum validation IoU**:

```python
checkpoint_callback = ModelCheckpoint(
    monitor='val_iou',
    mode='max',
    save_top_k=1
)
```

## Split Strategies

### Random Split (Baseline)

```python
from image_segmentation_pascal_voc.utils import multilabel_iterative_train_test_split

train_indices, val_indices = multilabel_iterative_train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
)
```

**Characteristics**:
- Uniform probability for each sample
- Stratified selection to preserve class proportions per split
- Standard practice in ML
- Simple and fast

### GoldSplitter (Smart Split)

The smart split is done from the class token of the Dinov3 ViT-S model.

```python
from image_classification_cifar10.utils import get_gold_splitter

gold_splitter = get_gold_splitter(cfg.gold_splitter)
split_table = gold_splitter.split_in_table(dataset)
splits = gold_splitter.get_split_indices(
    split_table, selection_key="selected", idx_key="idx"
)
train_indices = np.array(list(splits["train"]))
val_indices = np.array(list(splits["val"]))
```

**Characteristics**:
- Considers class labels for balanced splits
- Aims for optimal distribution
- May lead to more representative validation sets
- Potentially better generalization

### Evaluation Criteria

Compare the two methods on:
- **Convergence Speed**: Epochs to reach best performance
- **Stability**: Variance in validation metrics across epochs
- **Test Performance**: Final performance on held-out test set

## Viewing Results


After running the experiment, start the MLFlow UI:
```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser to compare results between split methods.
