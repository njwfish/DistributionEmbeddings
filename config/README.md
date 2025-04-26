# Configuration Structure

This directory contains the configuration files for the Distribution Embeddings project, organized according to Hydra's compositional configuration pattern.

## Directory Structure

- `config.yaml`: Main configuration file with default settings
- `experiment/`: Experiment-specific configurations
- `dataset/`: Dataset configurations
- `encoder/`: Encoder model configurations
- `model/`: Model configurations
- `generator/`: Generator configurations
- `optimizer/`: Optimizer configurations
- `scheduler/`: Learning rate scheduler configurations
- `training/`: Training configurations
- `mixer/`: Distribution mixer configurations
- `wandb/`: Weights & Biases logging configurations
- `hydra/`: Hydra-specific configurations

## Usage

To run an experiment, use the following command:

```bash
python main.py experiment=<experiment_name>
```

For example:

```bash
python main.py experiment=mnist_pca
```

## Creating New Experiments

To create a new experiment:

1. Create a new YAML file in the `experiment/` directory
2. Set the `@package _global_.experiment` at the top
3. Define experiment-specific parameters
4. Override default configurations as needed

Example:

```yaml
# @package _global_.experiment

# Experiment name and description
name: my_new_experiment
description: "Description of my new experiment"

# Override defaults from config.yaml
defaults:
  - /dataset: my_dataset

# Experiment-specific parameters
latent_dim: 16
hidden_dim: 64
batch_size: 32
lr: 1e-4
```

## Parameter Overrides

You can override specific parameters on the command line:

```bash
python main.py experiment=mnist_pca experiment.batch_size=32 training.max_epochs=100
```

This will run the `mnist_pca` experiment with a batch size of 32 and 100 epochs.

## Available Experiments

### MNIST Experiments
- `mnist_pca`: MNIST PCA experiment with distribution embeddings
- `mnist_multinomial`: MNIST with multinomial distribution embeddings
- `mnist_mixed`: MNIST mixed distributions experiment

### Synthetic Data Experiments
- `mvn`: Multivariate normal distribution experiment
- `gmm`: Gaussian Mixture Model experiment
- `normal`: Normal distribution experiment with Wasserstein generator
- `multinomial_fr`: Multinomial distribution with Fisher-Rao metric

### Biological Sequence Experiments
- `pfam`: Pfam protein families experiment with ESM and ProGen2
- `synthetic_protein`: Synthetic protein sequence generation
- `synthetic_dna`: Synthetic DNA sequence generation
- `methylation`: DNA methylation analysis
- `essential_genes`: Essential genes analysis

### Text Generation
- `pubmed`: PubMed NLP experiment with BERT and GPT-2

## Configuration Inheritance

The configuration system uses Hydra's inheritance mechanism. For example:

1. Base defaults are specified in `config.yaml`
2. Experiment-specific overrides are in `experiment/<name>.yaml`
3. Component configurations are in their respective directories

This allows for modular configuration where components can be mixed and matched. 