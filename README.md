# Generative Distribution Embeddings

Generative Distribution Embeddings (GDEs) are a framework that lifts autoencoders to the space of distributions. In GDEs, an encoder acts on _sets_ of samples, and the decoder is replaced by a generator which aims to match the input distribution. This repository contains implementations of several different GDE architectures, code to benchmark GDEs on synthetic distributions, and demonstrations of GDEs for several large-scale modelling problems in computational biology.

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

The project uses a hash-based output tracking system that organizes experiment outputs hierarchically:

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

2. **Experiment-specific files**:
   - `config.yaml`: The full Hydra configuration for the run
   - `best_model.pt`: Checkpoint of the model with the best validation performance
   - `checkpoint_epoch_N.pt`: Periodic checkpoints during training
   - `metrics.json`: Key metrics logged during training and evaluation
   - `wandb/`: If Weights & Biases is enabled, logs and metadata are synced here.

3. **Multirun outputs**:
   - When using `--multirun`, Hydra creates a parent directory for the multirun job
   - Individual run outputs are nested within this parent directory
   - Located in the `multirun/` directory by default

This structured output system facilitates:
- Easy tracking and comparison of experiments
- Reproducibility of results
- Efficient storage and retrieval of models and data

## Applications

This section details the main applications and experiments implemented in this project. Each application leverages the distribution embedding framework to model and generate data in various domains. For more in-depth theoretical background and results, a supplementary manuscript is available (see `generative_distribution_embeddings (7).pdf`).

### 1. Multinomial Distributions

*   **Description**: Models sets of multinomial distributions. This is a foundational statistical experiment to validate the core embedding and generation capabilities.
*   **Configuration**:
    *   Experiment: `config/experiment/multinomial.yaml` (name: `multinomial`)
    *   Dataset: `config/dataset/multinomial.yaml`
*   **Dataset**: `datasets.distribution_datasets.MultinomialDistributionDataset` (generates synthetic multinomial data).
*   **How to run**:
    ```bash
    python main.py experiment=multinomial
    ```
*   **Notebook**: `notebooks/multinomial_distributions.ipynb`

### 2. Multinomial MNIST

*   **Description**: Treats MNIST image pixel values (normalized and binned) as samples from multinomial distributions. Each image is a set of pixel distributions.
*   **Configuration**:
    *   Experiment: `config/experiment/mnist_multinomial.yaml` (name: `mnist_multinomial`)
    *   Dataset: `config/dataset/mnist.yaml`
    *   Mixer: `config/mixer/dirichlet_k.yaml` (often used)
*   **Dataset**: `datasets.mnist.MNISTDataset` (configured for multinomial interpretation).
*   **Mixer Class**: `mixer.mixer.SetMixer` with `dirichlet_k` config.
*   **How to run**:
    ```bash
    python main.py experiment=mnist_multinomial
    ```
*   **Notebooks**:
    *   `notebooks/mnist_multinomial.ipynb`
    *   `notebooks/mnist_multinomial_interpolation.ipynb`

### 3. Synthetic DNA

*   **Description**: Generates and models distributions of synthetic DNA sequences with repeating patterns or motifs. 
*   **Configuration**:
    *   Experiment (General): `config/experiment/synthetic_dna.yaml` (name: `synthetic_dna`)
    *   Experiment (Multinomial DNA): `config/experiment/dna_multinomial.yaml` (name: `dna_multinomial`)
    *   Dataset: `config/dataset/synthetic_dna.yaml`
*   **Dataset**: `datasets.synthetic_dna.SyntheticDNADataset`.
*   **How to run**:
    ```bash
    python main.py experiment=synthetic_dna
    # OR for the multinomial variant
    python main.py experiment=dna_multinomial
    ```
*   **Notebook**: `notebooks/dna_multinomial.ipynb`

### 4. Multivariate Normal (MVN) Distributions

*   **Description**: Models sets of Multivariate Normal (MVN) distributions. This includes experiments with systematic variations in model components.
*   **Configuration**:
    *   Main Experiment: `config/experiment/mvn.yaml` (name: `mvn`)
    *   Systematic Variations (examples):
        *   `config/experiment/mvn_sys_sw.yaml`
        *   `config/experiment/mvn_sys_vae.yaml`
        *   `config/experiment/mvn_sys_mmd.yaml`
        *   `config/experiment/mvn_sys_sinkhorn.yaml`
    *   Dataset: `config/dataset/mvn.yaml`
*   **Dataset**: `datasets.distribution_datasets.MVNDataset`.
*   **How to run**:
    ```bash
    python main.py experiment=mvn
    # For systematic variations, e.g.:
    # python main.py experiment=mvn_sys_vae
    ```
*   **Notebooks**:
    *   `notebooks/mvn_ot.ipynb`
    *   `notebooks/mvn_dists.ipynb`

### 5. Gaussian Mixture Models (GMM)

*   **Description**: Models sets of Gaussian Mixture Models. These experiments typically use an underlying MVN dataset and a Dirichlet mixer to create complex GMMs. Systematic variations explore different encoders and generators.
*   **Configuration**:
    *   Main Experiment: `config/experiment/gmm.yaml` (name: `gmm`)
    *   Systematic Variations (examples, often run via shell scripts like `sys.sh`, `sys2.sh`):
        *   `config/experiment/gmm_sys_vae.yaml`
        *   `config/experiment/gmm_sys_diff.yaml` 
        *   `config/experiment/gmm_sys_wormhole.yaml`
    *   Dataset: `config/dataset/mvn.yaml`
    *   Mixer: `config/mixer/dirichlet_k.yaml`
*   **Dataset**: `datasets.distribution_datasets.MVNDataset` (used with a mixer).
*   **Mixer Class**: `mixer.mixer.SetMixer` with `dirichlet_k` config.
*   **How to run**:
    ```bash
    python main.py experiment=gmm
    # Systematic experiments often launched via scripts like sys.sh / sys2.sh
    # e.g., python main.py experiment=gmm_sys_vae encoder=gnn
    ```
*   **Notebooks**:
    *   `notebooks/gmm_ot.ipynb`
    *   `notebooks/gmm_dists.ipynb`
    *   `notebooks/gmm_ot_pot_failure.ipynb`

### 6. Lineage Tracing (LT)

*   **Description**: Learns representations of clonal populations in lineage-traced single-cell RNA-seq data (e.g., from Weinreb et al., 2020). Models distributions of cell states over time.
*   **Configuration**:
    *   Experiment: `config/experiment/lineage_tracing.yaml` (name: `lineage`)
    *   Dataset: `config/dataset/lineage_tracing.yaml`
*   **Dataset**: `datasets.lineage_tracing.LTSeqDataset` (handles data download and processing).
*   **How to run**:
    ```bash
    python main.py experiment=lineage_tracing
    # or potentially using the config name:
    # python main.py experiment=lineage
    ```
*   **Notebook**: `notebooks/lineage_tracing_MI.ipynb` (focuses on mutual information analysis).

### 7. Perturbation Prediction (Perturb-seq / Essential Genes)

*   **Description**: Focuses on predicting transcriptomic changes (e.g., from Perturb-seq experiments like the "essential_genes" dataset) in response to genetic perturbations. Models distributions of gene expression profiles.
*   **Configuration**:
    *   Main: `config/experiment/essential_genes.yaml` (name: `essential_genes_exp`)
    *   Dataset: `config/dataset/essential_genes.yaml`
    *   Encoder Example: `config/encoder/resnet_pert_pred.yaml` (for `DistributionEncoderResNetPertPredictor`)
    *   Loss: `config/loss/perturbseq.yaml`
*   **Dataset**: `datasets.perturbseq_dataset.PerturbseqDataset`.
*   **How to run**:
    ```bash
    python main.py experiment=essential_genes
    ```

### 8. Optical Pooled Screening (OPS)

*   **Description**: Analyzes data from Optical Pooled Screening, a high-throughput functional genomics technique using microscopy. Involves modeling distributions of image-based cellular phenotypes. Some configurations (e.g., `ops_pert.yaml`) may incorporate perturbation information.
*   **Configuration**:
    *   Main: `config/experiment/ops.yaml` (name: `ops`)
    *   With Perturbation aspect: `config/experiment/ops_pert.yaml` (name: `ops`, but uses `encoder: conv_gnn_pert_pred` and `loss: perturbspatial`)
    *   Dataset: `config/dataset/ops.yaml`, `config/dataset/ops_small.yaml`, `config/dataset/ops32.yaml`
*   **Dataset**: `datasets.ops.OPSDataset` (and `datasets.ops32.OPS32Dataset`). Handles image tiles and associated perturbation information.
*   **How to run**:
    ```bash
    python main.py experiment=ops
    # For OPS with perturbation prediction aspects:
    # python main.py experiment=ops_pert 
    ```
*   **Notebook**: `notebooks/ops.ipynb`

### 9. DNA Methylation

*   **Description**: Models distributions of DNA methylation patterns from sequence data. Uses HyenaDNA for generation.
*   **Configuration**:
    *   Generative: `config/experiment/methylation.yaml` (name: `methyl_exp`)
    *   Classification: `config/experiment/methylation_class.yaml` (name: `methyl_exp` but with classification encoder/loss)
    *   Dataset: `config/dataset/methylation.yaml`
*   **Dataset**: `datasets.dna_dataset.DNADataset`.
*   **How to run**:
    ```bash
    python main.py experiment=methylation
    # For classification:
    # python main.py experiment=methylation_class loss=classification
    ```
*   **Notebooks**: `notebooks/methylation.ipynb`

### 10. GPRA (Gigantically Parallel Reporter Assay) DNA

*   **Description**: Learns expression patterns from Gigantically Parallel Reporter Assay (GPRA) DNA data. This involves modeling distributions of DNA sequences (e.g., promoters) associated with different expression level quantiles.
*   **Configuration**:
    *   Main: `config/experiment/gpra_dna.yaml` (name: `gpra_dna_exp`)
    *   VAE variant: `config/experiment/gpra_dna_vae.yaml`
    *   Ordered variant: `config/experiment/gpra_dna_ordered.yaml` (uses `loss: ordered`)
    *   Dataset: `config/dataset/gpra_dna.yaml`
*   **Dataset**: `datasets.gpra_dna_dataset.GPRADNADataset`.
*   **How to run**:
    ```bash
    python main.py experiment=gpra_dna
    # Or for variants:
    # python main.py experiment=gpra_dna_vae
    ```
*   **Notebook**: `notebooks/gpra.ipynb`

### 11. Viral Spike Protein Distributions

*   **Description**: Models distributions of viral spike protein sequences. Utilizes ESM (Evolutionary Scale Modeling) architectures for embeddings and ProGen2 for sequence generation. Data sourced from GISAID.
*   **Configuration**:
    *   Experiment: `config/experiment/virus.yaml` (name: `virus`)
    *   Dataset: `config/dataset/virus.yaml`
*   **Dataset**: `datasets.virus.ViralDataset`.
*   **How to run**:
    ```bash
    python main.py experiment=virus
    ```
*   **Notebook**: `notebooks/gisaid.ipynb` (for analysis and data handling related to viral sequences).

## Experiment Management

The project includes a command-line interface (`experiment_cli.py`) and utility functions (`utils/experiment_utils.py`) for managing and analyzing experiment results.

### Key Features

- **List experiments**: View all completed experiments with their configurations and key metrics.
- **Show experiment details**: Display the full configuration and results for a specific experiment.
- **Compare experiments**: Compare configurations and metrics between two or more experiments.
- **Load models and data**: Utilities to easily load trained models and datasets from experiment outputs.
- **Hashing and reproducibility**: Ensures that each experiment run with a unique configuration is stored in a separate, identifiable directory.

### CLI Usage Examples

```bash
# List all experiments (shows name, hash, key metrics)
python experiment_cli.py list

# Show detailed config and metrics for an experiment (use name or hash)
python experiment_cli.py show mvn_exp_a1b2c3d4

# Compare two experiments side-by-side
python experiment_cli.py compare mvn_exp_a1b2c3d4 mvn_exp_e5f6g7h8

# Filter experiments by name
python experiment_cli.py list --name_contains mvn

# Filter experiments by parameter values (e.g., latent_dim=64)
python experiment_cli.py list --param "experiment.latent_dim=64"
```

The CLI leverages the `utils/experiment_utils.py` module, which provides functions for parsing experiment configurations and results from the `outputs/` and `multirun/` directories.

## Model Architectures

The project implements several model architectures for encoders, decoders, and generators:

- **Encoders**:
  - `encoder.encoders.ResNetDistEncoder`: ResNet-based encoder for distribution embeddings.
  - `encoder.encoders.MLPDistEncoder`: MLP-based encoder.
  - `encoder.nlp_encoders.BertSetEncoder`: BERT-based encoder for sets of documents.
  - `encoder.conv_gnn.ConvGNNEncoder`: Graph Convolutional Network encoder.
  - `encoder.protein_encoders.ProteinSetEncoder`: ESM-based encoder for sets of protein sequences.
  - `encoder.dna_conv_encoder.DNAConvEncoder`: Convolutional encoder for DNA sequences.
- **Generators**:
  - `generator.ddpm.DDPM`: Denoising Diffusion Probabilistic Model.
  - `generator.gpt2_generator.GPT2Generator`: GPT-2 based text generator.
  - `generator.direct.DirectGenerator`: Simple direct generator (e.g., MLP).
  - `generator.cvae.CVAE`: Conditional Variational Autoencoder.
  - `generator.hyenadna_generator.HyenaDNAGenerator`: HyenaDNA for genomic sequence generation.
  - `generator.protein_generator.Progen2Generator`: ProGen2 for protein sequence generation.
- **Models (often used within CVAE or other frameworks)**:
  - `model.gnn.GNN`: Graph Neural Network.
  - `model.unet.UNet`: U-Net for diffusion models.
  - `model.vae_mlp.VAEMLP`: MLP-based VAE.

Refer to the respective configuration files in `config/encoder/`, `config/generator/`, and `config/model/` for detailed settings. 
