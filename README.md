# goldener-examples

Different examples using Goldener during training Machine Learning models.
These examples systematically compares goldener features with standard strategies for training machine learning models.

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

- [Goldener](https://github.com/goldener-data/goldener): Smart data orchestration strategies for training machine learning models
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

Goldener provides intelligent data orchestration strategies that aim to optimize the training process of machine learning models.
