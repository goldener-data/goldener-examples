# gold-split-examples

Different examples using Goldener GoldSplitter during training Machine Learning models.
These examples systematically compares two data splitting strategies for training machine learning models.
Data splitting is a critical step that can significantly impact model performance and generalization.
While random splitting is the standard approach, smart splitting strategies like GoldSplitter aim
to create more balanced and representative train/validation splits, potentially leading to:

- Better model generalization
- More reliable validation metrics
- Faster convergence
- More representative performance estimation


Thus, in all the examples, two data splitting strategies for creating training and validation datasets are compared:

1. **Random Split**: Traditional random split using scikit-learn
2. **Smart Split**: Intelligent split using GoldSplitter from the Goldener library

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install dependencies using:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra dev
uv run pre-commit install

# depending on the examples you want to run, you might need to install additional extras, for instance:
uv sync --extra vision
```

The pre-commit hooks will automatically run:
- **ruff**: Linting and formatting
- **mypy**: Type checking
- Additional checks: trailing whitespace, YAML validation, etc.

## Main dependencies

- [Goldener](https://github.com/goldener-data/goldener): Smart data splitting
- [scikit-learn](https://scikit-learn.org/): Random uniform data splitting
- [PyTorch](https://pytorch.org/): Neural networks building blocks/losses and data loading
- [Pytorch Lightning](https://www.pytorchlightning.ai/): Neural network training and evaluation framework
- [Torchmetrics](https://torchmetrics.readthedocs.io/): Model evaluation metrics
- [Mlflow](https://mlflow.org/): Experiment tracking
- [Hydra](https://hydra.cc/): Configuration management

## Examples

### 1. Image Classification: CIFAR-10 with different image classification models (ResNet, ViT, etc.)

**Quick Start**:
```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Run experiment
cd image_classification_cifar10
uv run python cifar10_experiment.py
```

See the [detailed README](image_classification_cifar10/README.md) for more information.

### 2. Image Segmentation: Pascal VOC with different segmentation models (Deeplab, FPN, etc.)

**Quick Start**:
```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Run experiment
cd image_segmentation_pascal_voc
uv run python voc_experiment.py
```

See the [detailed README](image_segmentation_pascal_voc/README.md) for more information.

### 3. Text Classification: IMDb Movie Reviews with CNN and BERT-Base

**Quick Start**:
```bash
# Install dependencies (from repo root)
uv sync --extra text

# Run experiment
cd text_classification_imdb
uv run python imdb_experiment.py
```

See the [detailed README](text_classification_imdb/README.md) for more information.


## About Goldener

Goldener provides intelligent data splitting strategies that aim to create
more balanced and representative train/validation splits compared to traditional random splitting.
