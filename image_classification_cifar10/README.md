# CIFAR-10 Split Comparison Experiment

This experiment applies different Goldener features during the training
of different image classification models on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). So far, illustrated Goldener features are:
- **GoldSplitter**: A smart data splitting method leveraging pretrained model to identify representative samples for training and validation sets.


## Table of Contents

- [Main Components](#main-components)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [Split Strategies](#split-strategies)
- [Viewing Results](#viewing-results)


## Main components

- **Configuration**: The experiment is configured from a config file loaded from Hydra 
for flexible configuration management. It allows to specify the hyperparameters and 
logging parameter for the model training/evaluation but as well all the Goldener settings.

- **CIFAR10DataModule**: A specific Pytorch Lightning Datamodule allowing to load data from the CIFAR-10 dataset from torchvision
(50,000 training images, 10,000 test images). Depending on the configuration, only a subset of the
training images is used for training/validation. Duplication of some samples is as well possible.

- **Cifar10LightningModule**: A specific Pytorch Lightning LightningModule allowing to train and evaluate different
image classification models (ResNet-18, ViT-S, custom CNN) for the CIFAR-p10 dataset.

- **Trainer**: PyTorch Lightning Trainer for efficient training management allowing to handle training, validation and
testing loops. It allows as well to checkpoint the best model based on validation AUROC metric.

- **Logging**: MLFlow for experiment tracking allowing to compare the training strategies based on the logged metrics.

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Make sure you're in the experiment directory
cd image_classification_cifar10

# Run experiment (uses default config)
uv run python cifar10_experiment.py

# View results
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.


## Technical Details

### Dataset: CIFAR-10

- **Classes**: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training samples**: 50,000 images (5000 per class)
- **Test samples**: 10,000 images
- **Image size**: initially 32×32 pixels but resized to 224×224, RGB color


### Training Configuration

- **Optimizer**: Adam
  - Learning rate: 0.001 (default, configurable)
  - Weight decay: 0 (no L2 regularization)

- **Loss Function**: CrossEntropyLoss
  - Applied to raw logits

- **Batch Size**: 256 (default, configurable)
- **Max Epochs**: 50 (default, configurable)

### Evaluation Metrics

**Primary Metric (for model selection)**:
- **AUROC (Area Under ROC Curve)**: Measures the model's ability to distinguish between classes
  - Task: Multiclass classification
  - Used for: Model checkpoint selection (best validation AUROC)

**Secondary Metrics**:
- **Accuracy**: Percentage of correct predictions

All metrics are computed and logged for both training and validation sets at each epoch.

### Model Selection

The best model is selected based on **maximum validation AUROC**:

```python
checkpoint_callback = ModelCheckpoint(
    monitor='val_auroc',
    mode='max',
    save_top_k=1
)
```

## Split Strategies

### Random Split (Baseline)

```python
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=targets,
)
```

**Characteristics**:
- Uniform probability for each sample
- Stratified selection to preserve class proportions per split
- Standard practice in ML
- Simple and fast

### GoldSplitter (Smart Split)

The smart split is done from the class token of the Dinov3 ViT-L model.

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
- Leverage pretrained feature to spot representative samples for training and validation sets
- Aims for optimal distribution
- Potentially leads to better generalization

## Viewing Results

After running the experiment, start the MLFlow UI:
```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser to compare results between run.
