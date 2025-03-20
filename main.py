import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
import wandb
import os

# Import our resolver for sum operations
import utils.config_resolvers as config_resolvers
import utils.hash_utils as hash_utils

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Compute config hash for reproducibility
    config_hash = hash_utils.hash_config(cfg)
    logger.info(f"Configuration hash: {config_hash}")
    
    # Check if we have already run this experiment
    original_cwd = hydra.utils.get_original_cwd()
    base_output_dir = os.path.join(original_cwd, "outputs")
    existing_dir = hash_utils.find_matching_output_dir(cfg, base_dir=base_output_dir)
    
    if existing_dir is not None:
        logger.info(f"Found existing results for this configuration: {existing_dir}")
        # You could choose to exit here or continue with the same directory
    
    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        
    # Initialize W&B
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode
    )
    
    # Log config hash to W&B
    if wandb.run is not None:
        wandb.run.summary["config_hash"] = config_hash
    
    try:
        # Create the dataset
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataloader = DataLoader(dataset, batch_size=cfg.experiment.batch_size, shuffle=True)
        
        # Create encoder
        encoder = hydra.utils.instantiate(cfg.encoder)
        
        # Create generator (with model already instantiated)
        generator = hydra.utils.instantiate(cfg.generator)
        
        # Get model parameters
        model_parameters = list(encoder.parameters()) + list(generator.model.parameters())
        
        # Create optimizer and scheduler
        optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model_parameters)
        scheduler = hydra.utils.instantiate(cfg.scheduler)(optimizer=optimizer)
        
        # Create trainer
        trainer = hydra.utils.instantiate(cfg.training)
        
        # Run training with the hash-based output directory
        stats = trainer.train(
            encoder=encoder,
            generator=generator,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=base_output_dir,
            config=cfg,
        )
        
        logger.info(f"Training completed. Best epoch: {stats['best_epoch']}")
        
        # Generate some samples with the trained model
        samples = trainer.generate_samples(encoder, generator, dataloader)
        logger.info(f"Generated samples shape: {samples['generated'].shape}")
        
        # Log generated samples to W&B (optional)
        if wandb.run is not None and samples is not None:
            # We'll log just a few samples to avoid excessive data transfer
            n_examples = min(5, samples['original'].shape[0])
            
            # For 1D data
            if len(samples['original'].shape) == 3:  # [batch, set_size, features]
                for i in range(n_examples):
                    wandb.log({
                        f"samples/original_{i}": wandb.Histogram(samples['original'][i].numpy()),
                        f"samples/generated_{i}": wandb.Histogram(samples['generated'][i].numpy())
                    })
            # For image data
            elif len(samples['original'].shape) == 4:  # [batch, set_size, height, width]
                # Log a grid of images
                orig_images = samples['original'][:n_examples].reshape(-1, *samples['original'].shape[2:])
                gen_images = samples['generated'][:n_examples].reshape(-1, *samples['generated'].shape[2:])
                
                wandb.log({
                    "samples/original": [wandb.Image(img) for img in orig_images],
                    "samples/generated": [wandb.Image(img) for img in gen_images]
                })
    
    finally:
        # Make sure to finish the W&B run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 