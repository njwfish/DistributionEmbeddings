# Distribution Embeddings

This repository contains implementation of Distribution Embeddings for various statistical distributions, including normal, Poisson, and multinomial distributions. The project provides tools for embedding statistical distributions into a latent space, allowing for efficient manipulation and generation of distributions.

## Project Structure

```
.
├── config/                      # Main Hydra configuration files
├── datasets/                    # Dataset implementations
│   ├── distribution_datasets.py # Distribution datasets
│   ├── mnist.py                 # MNIST dataset utilities
│   └── pubmed.py                # PubMed abstracts dataset
├── encoder/                     # Encoder models
│   ├── encoders.py              # Various encoder implementations
│   └── nlp_encoders.py          # NLP-specific encoders (BERT)
├── generator/                   # Generator models and losses
│   ├── generators.py            # Generator implementations
│   ├── gpt2_generator.py        # GPT-2 based text generator
│   ├── ddpm.py                  # Diffusion models
│   └── losses.py                # Loss functions for generators
├── notebooks/                   # Jupyter notebooks for examples and experiments
├── utils/                       # Utility functions
│   ├── hash_utils.py            # Utilities for config hashing and output tracking
│   ├── experiment_utils.py      # Experiment management utilities
│   └── plot_utils.py            # Plotting utilities
├── experiment_cli.py            # CLI tool for experiment management
├── layers.py                    # Neural network layers
├── main.py                      # Main training script
├── training.py                  # Training implementation
└── README.md                    # This file
```

## Usage

### Training a Model

To train a model, run the `main.py` script with Hydra configuration options:

```bash
# Train with default configuration
python main.py

# Change training parameters
python main.py training.num_epochs=200 training.early_stopping=False

# Multirun with different seeds
python main.py --multirun seed=42,43,44,45,46
```

### NLP Document Distribution Example

This project includes an NLP example for modeling document distributions using BERT and GPT-2. The example uses the PubMed dataset, organizing documents by MeSH (Medical Subject Heading) tags to form document sets.

#### Architecture

The NLP document distribution model consists of:

1. **BERT Document Encoder**: Processes each document using BERT and extracts document features
2. **Distribution Encoder**: Encodes the set of document features into a distribution embedding
3. **GPT-2 Generator**: Generates new text samples conditioned on the distribution embedding

#### Running the PubMed Example

To train the PubMed document distribution model:

```bash
# Train with default PubMed configuration
python main.py -c pubmed_nlp

# Customize BERT and GPT-2 settings
python main.py -c pubmed_nlp experiment.bert_model_name=bert-base-uncased experiment.freeze_bert=false
```

The model learns to:
- Embed sets of documents (sharing the same MeSH tag) into a common latent space
- Generate new documents that reflect the distribution of documents in the original set
- Capture the semantic characteristics of document sets

#### Customizing the NLP Model

You can customize various aspects of the NLP model:

- **BERT model**: Change the base BERT model or its output dimensions
  ```bash
  python main.py -c pubmed_nlp experiment.bert_model_name=bert-large-uncased experiment.bert_output_dim=1024
  ```

- **Distribution encoder**: Try different distribution encoder architectures
  ```bash
  python main.py -c pubmed_nlp experiment.distribution_encoder_type=gnn
  ```

- **GPT-2 settings**: Adjust the generator's conditioning method or parameters
  ```bash
  python main.py -c pubmed_nlp experiment.condition_method=additive experiment.freeze_gpt2=false
  ```

### Experiment Management

The project includes a hash-based output tracking system that organizes experiment outputs based on configuration hashes, making it easy to track and compare experiments.

#### Experiment CLI Tool

Use the `experiment_cli.py` script to manage and explore your experiments:

```bash
# List all experiments
./experiment_cli.py list

# Show details for a specific experiment (using name or hash)
./experiment_cli.py show experiment_name_f7c3a9b42d

# Compare two experiments
./experiment_cli.py compare experiment1 experiment2

# Find experiments matching a config file
./experiment_cli.py find path/to/config.yaml

# Calculate hash for a config file
./experiment_cli.py hash path/to/config.yaml
```

#### Hash-based Output Directories

Experiment outputs are organized in directories named with the experiment name and a unique configuration hash:

```
outputs/
├── normal_f7c3a9b42d/     # Experiment with normal distributions
│   ├── config.yaml        # Saved configuration
│   ├── best_model.pt      # Best model checkpoint
│   └── checkpoint_*.pt    # Training checkpoints
├── mnist_a1b2c3d4e5/      # Experiment with MNIST data
│   └── ...
└── pubmed_7e8f9g0h1i/     # Experiment with PubMed document distributions
    └── ...
```

This system provides several benefits:
- Automatic organization of experiment outputs
- Easy identification of duplicate experiments
- Resumption of training from previous runs
- Simple experiment comparison

### Visualization

To visualize the results, run the `visualize.py` script:

```bash
# Visualize results from the last trained model
python visualize.py

# Visualize specific model results
python visualize.py model_path=/path/to/model/checkpoint
```

### Notebooks

The `notebooks/` directory contains Jupyter notebooks with examples and experiments to help understand the project:

- Data exploration
- Model training and evaluation
- Visualization of embeddings
- Examples of distribution manipulation

## Configuration

The project uses Hydra for configuration management. The main configuration groups are:

- **dataset**: Configuration for data generation (distribution type, parameters, etc.)
- **encoder**: Configuration for the encoder architecture
- **generator**: Configuration for the generator architecture
- **training**: Configuration for training parameters (learning rate, epochs, loss function, etc.)
- **optimizer**: Configuration for the optimizer
- **scheduler**: Configuration for the learning rate scheduler
- **wandb**: Configuration for Weights & Biases logging

See the `config/` directory for detailed configuration options.

## Experiment Reproducibility

The hash-based output system ensures experiment reproducibility by:

1. Generating a unique hash for each configuration
2. Storing outputs in deterministically named directories
3. Saving the full configuration alongside results
4. Automatically detecting and reusing existing experiment outputs
5. Supporting training resumption from checkpoints

When you run an experiment with a configuration that matches a previous run, the system will detect this and can either:
- Continue training from the last checkpoint
- Inform you that results already exist for this configuration

## Weights & Biases Integration

This project supports logging to [Weights & Biases](https://wandb.ai/). Configure your W&B settings in the config file:

```yaml
wandb:
  project: "distribution-embeddings"
  entity: "your-username"
  mode: "online"  # Set to "disabled" to disable W&B logging
```

The configuration hash is also logged to W&B, making it easy to correlate W&B runs with local experiment directories.

## License

[MIT License](LICENSE) 