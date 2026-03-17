# Text Classification: IMDb Movie Reviews

This example compares two data splitting strategies for binary sentiment classification on the
[IMDb Large Movie Review Dataset](https://huggingface.co/datasets/imdb).

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

- **IMDbDataModule**: A specific Pytorch Lightning DataModule allowing to load data from the
IMDb Large Movie Review dataset (50k movie reviews). Depending on the configuration, only a subset of the
training reviews is used for training/validation. Duplication of some samples is as well possible.

- **IMDbLightningModule**: A specific Pytorch Lightning LightningModule allowing to train and evaluate different
text classification models (custom CNN, Bert-based classifier) for the IMDb Large Movie Review dataset.

- **Trainer**: PyTorch Lightning Trainer for efficient training management allowing to handle training, validation and
testing loops. It checkpoints the best model based on validation IoU metric.

- **Logging**: MLFlow for experiment tracking allowing to compare the different splitting strategies based on the logged metrics.

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync --extra text

# Make sure you're in the experiment directory
cd text_classification_imdb
# Run both split methods (uses default config)
uv run python imdb_experiment.py

# View results
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## Technical Details

### Dataset: Pascal VOC 2012

- **Task**: Binary sentiment classification (positive vs negative reviews)
- **Classes**: 2 (positive, negative)
- **Training samples**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Test samples**: 25,000 reviews (12,500 positive, 12,500 negative)
- **Input**: raw text reviews (tokenized and converted to token IDs for model input)

### Training Configuration

- **Optimizer**: AdamW
  - Learning rate: 0.001 (default, configurable)
  - Weight decay: 0 (no L2 regularization)

- **Loss Function**: CrossEntropyLoss for binary classification

- **Batch Size**: 32 (default, configurable)
- **Max Epochs**: 10 (default, configurable)

### Evaluation Metrics

**Primary Metric (for model selection)**:
- **AUROC (Area Under the ROC curve)**: measures the model's ability to distinguish between positive and negative reviews
  - Task: Binary classification
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
