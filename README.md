# Distribution Embeddings

This repository contains implementation of Distribution Embeddings for various statistical distributions, including normal, Poisson, and multinomial distributions. The project provides tools for embedding statistical distributions into a latent space, allowing for efficient manipulation and generation of distributions.

## Setup and Configuration

### Project Structure

```
.
├── config/                      # Main Hydra configuration files
│   ├── dataset/                 # Dataset configurations
│   ├── encoder/                 # Encoder model configurations
│   ├── generator/               # Generator configurations
│   ├── model/                   # Model architecture configurations
│   ├── optimizer/               # Optimizer configurations
│   ├── scheduler/               # Learning rate scheduler configurations
│   ├── training/                # Training configurations
│   ├── wandb/                   # Weights & Biases configurations
│   ├── hydra/                   # Hydra-specific configurations (incl. Slurm)
│   ├── experiment/              # Experiment-specific configurations
│   │   ├── mvn.yaml             # Multivariate normal distribution experiment
│   │   ├── gmm.yaml             # Gaussian Mixture Model experiment
│   │   ├── normal.yaml          # Normal distribution experiment
│   │   ├── pubmed.yaml          # PubMed NLP experiment
│   │   ├── essential_genes.yaml # Essential genes experiment
│   │   └── ...                  # Other experiment configurations
│   └── config.yaml              # Base configuration file
├── datasets/                    # Dataset implementations
│   ├── distribution_datasets.py # Statistical distribution datasets
│   ├── mnist.py                 # MNIST dataset utilities
│   ├── pubmed.py                # PubMed abstracts dataset
│   └── essential_genes_dataset.py # Essential genes dataset
├── encoder/                     # Encoder models
│   ├── encoders.py              # Various encoder implementations
│   ├── nlp_encoders.py          # NLP-specific encoders (BERT)
│   └── conv_gnn.py              # Graph convolutional network encoder
├── generator/                   # Generator models and losses
│   ├── losses.py                # Loss functions for generators
│   ├── ddpm.py                  # Diffusion models
│   ├── gpt2_generator.py        # GPT-2 based text generator
│   └── direct.py                # Direct generator implementation
├── model/                       # Model architectures
│   ├── gnn.py                   # Graph neural network models
│   └── unet.py                  # U-Net architecture for diffusion models
├── mixer/                       # Data mixing strategies
├── utils/                       # Utility functions
│   ├── hash_utils.py            # Utilities for config hashing and output tracking
│   ├── experiment_utils.py      # Experiment management utilities
│   ├── plot_utils.py            # Plotting utilities
│   └── visualization.py         # Visualization functions for results
├── notebooks/                   # Jupyter notebooks for examples and experiments
├── outputs/                     # Experiment outputs directory
├── data/                        # Data storage directory
├── multirun/                    # Directory for Hydra multirun output
├── layers.py                    # Neural network layer implementations
├── main.py                      # Main training script
├── training.py                  # Training implementation
├── experiment_cli.py            # CLI tool for experiment management
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

### Configuration System

This project uses [Hydra](https://hydra.cc/) for configuration management. The configuration is organized into groups:

- **experiment**: High-level experiment configurations that combine other config groups
- **dataset**: Data generation settings (distribution type, parameters, datasets)
- **encoder**: Encoder architecture configuration
- **generator**: Generator architecture configuration
- **model**: Model architecture settings (UNet, GNN, etc.)
- **training**: Training parameters (learning rate, epochs, loss function)
- **optimizer**: Optimizer configuration (Adam, SGD, etc.)
- **scheduler**: Learning rate scheduler configuration
- **mixer**: Data mixing strategies for distribution datasets
- **wandb**: Weights & Biases logging settings
- **hydra**: Hydra-specific configurations (including Slurm launcher)

The configuration system follows Hydra's compositional pattern:

1. Base defaults are specified in `config/config.yaml`
2. Experiment-specific configurations are in `config/experiment/*.yaml`
3. Component configurations are in their respective directories (e.g., `config/encoder/`)

Each experiment configuration (`config/experiment/*.yaml`) typically:
- Sets the experiment name and description
- Overrides default configurations for dataset, encoder, model, etc.
- Defines experiment-specific parameters like latent dimensions, batch size, etc.

For example, a typical experiment config looks like:

```yaml
# @package _global_.experiment

# Experiment name and description
name: mvn_exp
description: "Multivariate normal distribution experiment"

# Override defaults from config.yaml
defaults:
  - /dataset: mvn
  - /encoder: resnet
  - /model: diffusion_mlp
  - /generator: ddpm

# Experiment-specific parameters
latent_dim: 64
hidden_dim: 128
set_size: 1000
batch_size: 256
lr: 0.0002
```

All configuration files are stored in the `config/` directory with a modular structure that allows easy composition and overriding.

### Installation and Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch and torchvision
- NumPy and matplotlib
- Hydra for configuration management
- Weights & Biases for experiment tracking
- Transformers library for NLP models (BERT and GPT-2)
- scikit-learn and pandas for data processing
- Submitit for Slurm job submission

## Usage

### Basic Usage

The project uses [Hydra](https://hydra.cc/) for configuration management. At its simplest, you can run experiments with:

```bash
# Basic usage with default configuration
python main.py

# Override specific parameters
python main.py training.num_epochs=200 optimizer.lr=0.001

# Use a specific configuration file
python main.py --config-name pubmed_nlp
```

### Advanced Usage

For more advanced usage, you can:

```bash
# Override nested parameters
python main.py dataset.params.mean=0.0 dataset.params.std=1.0

# Use different config groups
python main.py dataset=normal encoder=mlp generator=mlp
```

### Multirun Mode and Slurm Integration

The project has built-in support for running multiple experiments and scaling to a Slurm cluster:

#### Local Multirun

For running multiple configurations locally:

```bash
# Simple multirun with different seeds
python main.py --multirun seed=42,43,44,45,46

# Grid search over multiple parameters
python main.py --multirun dataset=normal,poisson,multinomial encoder=mlp,transformer
```

#### Running on Slurm Cluster

For large-scale experiments, the project uses Hydra's Submitit launcher to submit jobs to a Slurm cluster:

```bash
# Run a multirun experiment on Slurm
python main.py --multirun hydra/launcher=slurm dataset=normal,poisson,multinomial
```

The Slurm configuration can be found in `config/hydra/launcher/slurm.yaml`. Default settings:

```yaml
partition: ou_bcs_low           # Slurm partition
gpus_per_node: 1                # Number of GPUs per node
cpus_per_task: 5                # CPUs per task
mem_gb: 256                     # Memory per node (GB)
timeout_min: 720                # Timeout (minutes)
array_parallelism: 8            # Maximum concurrent jobs
```

To customize Slurm settings for a specific run:

```bash
# Customize Slurm settings
python main.py --multirun \
  hydra/launcher=slurm \
  hydra.launcher.partition=your_partition \
  hydra.launcher.gpus_per_node=2 \
  hydra.launcher.timeout_min=1440 \
  hydra.launcher.mem_gb=128 \
  experiment=gmm
```


### Experiment Management CLI

The `experiment_cli.py` script provides a powerful command-line interface for managing experiments:

```bash
# List all experiments with key metadata
python experiment_cli.py list

# Show detailed information about a specific experiment
python experiment_cli.py show experiment_name_f7c3a9b42d

# Compare two experiments
python experiment_cli.py compare experiment1 experiment2
```

For more details on the experiment management system, see the "Experiment Management" section below.

### Output Structure

The project uses a sophisticated hash-based output tracking system that organizes experiment outputs hierarchically:

```
outputs/
└── experiment_name_[hash]/       # Hash-based experiment directories
    ├── config.yaml               # Complete experiment configuration 
    ├── best_model.pt             # Best model checkpoint
    ├── checkpoint_*.pt           # Training checkpoints at different epochs
    ├── metrics.json              # Evaluation metrics

```

The output system has several layers of organization:

1. **Hash-based experiment directories**: 
   - Each unique configuration gets a deterministic hash (MD5 of sorted config)
   - Directory names follow the pattern `experiment_name_[hash]`
   - Enables exact experiment reproducibility and automatic resumption
   - Stored in the top level of the `outputs/` directory

2. **Timestamped run directories**:
   - For experiments that don't use the hash-based system
   - Organized by date and time of execution
   - Contain all files related to a single run
   - Used primarily for exploratory experiments 

3. **Multirun directories**:
   - Created when running multiple configurations with `--multirun`
   - Each subfolder represents a separate run with different parameters
   - Organized hierarchically by date, time, and run number
   - Located in the separate `multirun/` directory at the project root

#### Hash Generation Process

The hash generation is controlled by `utils/hash_utils.py` and works as follows:

1. Configuration is converted to a dictionary and normalized
2. Non-deterministic keys are excluded (`hydra`, `seed`, `device`, `wandb`, etc.)
3. Keys are sorted alphabetically for consistent ordering
4. The configuration is serialized to JSON and hashed with MD5
5. The resulting hash becomes part of the output directory name

#### Experiment Resumption

When you run an experiment, the system:

1. Computes the hash of your current configuration
2. Searches for existing directories with matching hashes
3. If found, loads the latest checkpoint and resumes training
4. If not found, creates a new directory and starts fresh

This enables automatic experiment resumption without manual intervention:

```bash
# Run an experiment
python main.py --config-name pubmed_nlp

# If interrupted, simply run the same command again to resume
python main.py --config-name pubmed_nlp
```

The trainer will locate the existing directory using the configuration hash, load the latest checkpoint, and continue training from where it left off.

## Experiment Types

The project contains a variety of experiment configurations in the `config/experiment/` directory:

### Statistical Distribution Experiments

#### Multivariate Normal Distribution (MVN)

```bash
# Run the multivariate normal distribution experiment
python main.py experiment=mvn

# Customize MVN experiment parameters
python main.py experiment=mvn experiment.latent_dim=32 experiment.set_size=500
```

The MVN experiment models multivariate normal distributions using:
- Resnet-based distribution encoder
- Diffusion model (DDPM) generator
- MLP-based model architecture

#### Gaussian Mixture Model (GMM)

```bash
# Run the Gaussian mixture model experiment
python main.py experiment=gmm

# Customize GMM parameters
python main.py experiment=gmm experiment.n_mix=5 experiment.alpha=0.8
```

The GMM experiment uses a Dirichlet mixer to create mixtures of Gaussian distributions.

#### Normal Distribution with Wasserstein Generator

```bash
# Run normal distribution experiment with Wasserstein distance
python main.py experiment=normal

# Customize normal distribution experiment
python main.py experiment=normal experiment.hidden_dim=128 generator=wasserstein
```

#### Multinomial with Fisher-Rao Metric

```bash
# Run multinomial distribution with Fisher-Rao metric
python main.py experiment=multinomial_fr
```

### Computer Vision Experiments

#### MNIST PCA Experiment

```bash
# Run MNIST PCA experiment
python main.py experiment=mnist_pca
```

Models the distribution of MNIST digits using PCA-based dimensionality reduction.

#### MNIST Multinomial

```bash
# Run MNIST multinomial experiment
python main.py experiment=mnist_multinomial
```

Treats MNIST data as multinomial distributions for modeling.

#### CIFAR-10 and Fashion MNIST

```bash
# Run CIFAR-10 experiment
python main.py experiment=cifar10

# Run Fashion MNIST experiment
python main.py experiment=fashion_mnist
```

Both provide image distribution embedding experiments with different datasets.

### Biological Sequence Experiments

#### Pfam Protein Families

```bash
# Run Pfam protein families experiment
python main.py experiment=pfam
```

Uses protein language models (ESM and ProGen2) for protein family distribution modeling.

#### Synthetic DNA and Protein Generation

```bash
# Run synthetic protein sequence generation
python main.py experiment=synthetic_protein

# Run synthetic DNA sequence generation
python main.py experiment=synthetic_dna
```

Generate synthetic biological sequences from distribution embeddings.

#### DNA Methylation Analysis

```bash
# Run DNA methylation analysis
python main.py experiment=methylation
```

Models the distribution of DNA methylation patterns.

#### Essential Genes Analysis

```bash
# Run essential genes analysis
python main.py experiment=essential_genes

# Customize essential genes experiment
python main.py experiment=essential_genes generator=ddpm model=diffusion_gnn
```

Uses a specialized graph neural network for modeling essential gene patterns, bypassing the typical information bottleneck in single-cell gene expression modeling.

### Natural Language Processing Experiments

#### PubMed Document Distribution

```bash
# Run PubMed NLP experiment
python main.py experiment=pubmed

# Customize BERT and GPT-2 settings
python main.py experiment=pubmed experiment.bert_model_name=bert-base-uncased experiment.freeze_bert=false
```

The PubMed experiment embeds and generates scientific abstracts using:
- BERT-based document encoder
- Distribution encoder to capture document set characteristics
- GPT-2 generator to produce new documents

The NLP experiment architecture:
1. **BERT Document Encoder**: Processes each document and extracts features
2. **Distribution Encoder**: Encodes the set of document features
3. **GPT-2 Generator**: Generates new documents from the distribution embedding

## Experiment Management

The project includes a comprehensive experiment management system to help track, compare, and analyze experiments. This system is built around the concept of configuration hashing and organized output directories.

### Experiment CLI Tool

The `experiment_cli.py` script provides a powerful command-line interface for managing experiments:

```bash
# List all experiments with key metadata (encoder, generator, model, dataset)
python experiment_cli.py list

# Sort experiments by a specific field
python experiment_cli.py list --sort-by best_loss

# Show detailed information about a specific experiment (using name or hash)
python experiment_cli.py show experiment_name_f7c3a9b42d

# Compare two experiments to see configuration and performance differences
python experiment_cli.py compare experiment1 experiment2

# Find experiments matching a configuration file
python experiment_cli.py find path/to/config.yaml

# Calculate hash for a configuration file
python experiment_cli.py hash path/to/config.yaml

# Create metrics.json files to avoid loading checkpoints in the future
python experiment_cli.py create-metrics

# Clean up dead experiments (those with only config files and no outputs)
python experiment_cli.py cleanup --dry-run
python experiment_cli.py cleanup --force
```

#### CLI Features and Implementation

The experiment CLI tool offers several key features:

1. **Experiment Listing (`list`)**: 
   - Displays all experiments in a tabular format
   - Shows key model components (encoder, generator, model, dataset) by default
   - Never loads checkpoints for listing to ensure fast performance
   - Supports custom column selection and sorting
   - Filters and formats output for easy readability

2. **Experiment Details (`show`)**: 
   - Shows comprehensive information about a specific experiment
   - Displays detailed information about model components directly from the configuration
   - Shows important parameters for each component (encoder, generator, model, dataset)
   - Provides training information and available checkpoints
   - Optionally displays the full configuration in YAML or JSON format
   - Includes debug option for troubleshooting

3. **Experiment Comparison (`compare`)**: 
   - Side-by-side comparison of two experiment configurations
   - Highlights differences in model components and hyperparameters
   - Calculates percentage differences in performance metrics
   - Provides detailed configuration differences when requested

4. **Configuration-based Search (`find`)**: 
   - Finds experiments that match a given configuration file
   - Supports both exact and partial matching
   - Helps avoid duplicate experiments

5. **Hash Calculation (`hash`)**: 
   - Computes the deterministic hash for any configuration
   - Checks if an experiment with that hash already exists

6. **Metrics File Creation (`create-metrics`)**: 
   - Creates metrics.json files with performance data
   - Avoids the need to load checkpoints when viewing experiment information
   - Improves performance of the CLI tool

7. **Dead Experiment Cleanup (`cleanup`)**: 
   - Identifies experiments that only have config files without any outputs
   - Supports dry-run mode to preview changes before execution
   - Can move dead experiments to an archive directory instead of deleting
   - Requires confirmation before making changes

The CLI tool is designed to be efficient, never loading checkpoints unnecessarily and using cached metrics where possible to ensure fast performance even with large numbers of experiments.

### Experiment Metadata and Tracking

The experiment management system maintains various metadata files within each experiment directory:

1. **Configuration Files**:
   - `config.yaml`: Complete configuration used for the experiment
   - `.hydra/overrides.yaml`: Command-line arguments that overrode the base config

2. **Performance Metrics**:
   - `metrics.json`: Summary of key metrics (best loss, best epoch) for quick access
   - Log files with detailed training history

3. **Checkpoints**:
   - `best_model.ckpt`: Best performing model based on validation metrics
   - `*.ckpt`: Periodic checkpoints at regular intervals

The metadata and tracking system provides:
- Complete reproducibility through saved configurations
- Performance history for analysis
- Easy comparison between different approaches
- Efficient organization of hundreds of experiments

### Experiment Reproducibility

The hash-based output system ensures experiment reproducibility by:

1. Generating a unique hash for each configuration
2. Storing outputs in deterministically named directories
3. Saving the full configuration alongside results
4. Automatically detecting and reusing existing experiment outputs
5. Supporting training resumption from checkpoints

When you run an experiment with a configuration that matches a previous run, the system will detect this and can either:
- Continue training from the last checkpoint
- Inform you that results already exist for this configuration

### Training Resumption System

The project includes an automatic training resumption system that:

1. Identifies interrupted experiments using their configuration hash
2. Loads the latest checkpoint from the matching directory
3. Restores model parameters, optimizer state, and scheduler state
4. Continues training from the exact point where it stopped

This provides several benefits:
- Resilience against unexpected interruptions
- Efficient use of computational resources
- Simplified workflow when running long experiments
- Consistent results regardless of interruptions

To manually resume a specific experiment:

```bash
# Find the experiment you want to resume
python experiment_cli.py list

# Resume training with the same configuration
python main.py --config-name your_config_name
```

The system automatically handles the rest, identifying the matching experiment and continuing from the latest checkpoint.

## Visualization and Analysis

### Results Visualization

The project provides visualization utilities in the `utils/visualization.py` file. You can visualize results by:

1. Direct visualization during training:
   - The trainer automatically visualizes sample results at evaluation intervals
   - Visualizations are saved in the experiment output directory

2. Examining generated samples:
   - For image data (MNIST): Compare original and generated images
   - For distribution data: Compare original and generated distributions
   - For text data: Compare original and generated texts

3. Using visualization functions directly:
   ```python
   from utils.visualization import visualize_data, visualize_text_data
   
   # Visualize distribution data
   visualize_data(save_path="output.png", real=real_samples, generated=gen_samples)
   
   # Visualize text data
   visualize_text_data(output_dir="text_output", original_texts=orig_texts, generated_texts=gen_texts)
   ```

### Notebooks

The `notebooks/` directory contains Jupyter notebooks with examples and experiments:

- Data exploration for different distribution types
- Training examples and parameter tuning
- Visualization of embeddings and generated samples
- Distribution manipulation in latent space

## Weights & Biases Integration

This project supports logging to [Weights & Biases](https://wandb.ai/). Configure your W&B settings in the config file:

```yaml
wandb:
  project: "distribution-embeddings"
  entity: "your-username"
  mode: "online"  # Set to "disabled" to disable W&B logging
```

To enable W&B logging from the command line:

```bash
python main.py wandb.mode=online wandb.project=my-project wandb.entity=my-username
```

The configuration hash is also logged to W&B, making it easy to correlate W&B runs with local experiment directories.

## License

[MIT License](LICENSE) 