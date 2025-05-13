import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import shutil
from tqdm import tqdm
import logging
import wandb
from utils.hash_utils import get_output_dir, find_matching_output_dir
from utils.visualization import visualize_data, visualize_text_data
from omegaconf import OmegaConf, DictConfig

class Trainer:
    def __init__(
        self,
        num_epochs=100,
        log_interval=10,
        save_interval=20,
        eval_interval=5,
        sub_epoch_interval=1_000,
        early_stopping=True,
        patience=10,
        use_tqdm=True,
        mask_context_prob=0.0,
        sub_epoch=None
    ):
        """
        Initialize the trainer.
        
        Args:
            num_epochs: Number of epochs to train for
            log_interval: How often to log training metrics (in batches)
            save_interval: How often to save model checkpoints (in epochs)
            eval_interval: How often to run evaluation (in epochs)
            early_stopping: Whether to use early stopping
            patience: Number of evaluations with no improvement before early stopping
            use_tqdm: Whether to use tqdm progress bars
        """
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.sub_epoch_interval = sub_epoch_interval
        if sub_epoch is None:
            self.sub_epoch = save_interval == 1
        else:
            self.sub_epoch = sub_epoch
        self.early_stopping = early_stopping
        self.patience = patience
        self.use_tqdm = use_tqdm
        self.mask_context_prob = mask_context_prob
        
        self.logger = logging.getLogger(__name__)
        self.best_loss = float('inf')
        self.no_improve_count = 0
        # log sub_epoch_interval
        self.logger.info(f"Sub epoch interval: {self.sub_epoch_interval}, save interval: {self.save_interval}, eval interval: {self.eval_interval}")
    
    def _find_similar_experiment_by_name(self, experiment_name, current_config, base_dir):
        """
        Find experiments with the same name but different epoch counts.
        
        Args:
            experiment_name: Name of the experiment (without hash)
            current_config: Current configuration
            base_dir: Base output directory
            
        Returns:
            List of tuples (directory, num_epochs) of matching experiments
        """
        matching_experiments = []
        
        # Extract current num_epochs from config
        current_num_epochs = None
        if isinstance(current_config, DictConfig):
            config_dict = OmegaConf.to_container(current_config, resolve=True)
        else:
            config_dict = current_config.copy()
        
        current_num_epochs = config_dict['training']['num_epochs']

        config_dict.pop('training')

        # Look for directories with the same experiment name prefix
        for item in os.listdir(base_dir):
            if not os.path.isdir(os.path.join(base_dir, item)):
                continue
                
            # Check if this directory has the same experiment name prefix
            if item.startswith(f"{experiment_name}_"):
                exp_dir = os.path.join(base_dir, item)
                config_path = os.path.join(exp_dir, 'config.yaml')
                
                if not os.path.exists(config_path):
                    continue
                    
                try:
                    # Load the experiment config
                    exp_config = OmegaConf.load(config_path)
                    exp_config_dict = OmegaConf.to_container(exp_config, resolve=True)
                    
                    # Check if this experiment has a different num_epochs
                    exp_num_epochs = exp_config_dict['training']['num_epochs']
                    
                    if exp_num_epochs != current_num_epochs:
                        exp_config_dict.pop('training')
                        
                        # Compare the configs without epochs
                        if self._configs_match(config_dict, exp_config_dict):
                            self.logger.info(f"Found similar experiment: {item} with {exp_num_epochs} epochs")
                            matching_experiments.append((exp_dir, exp_num_epochs))
                except Exception as e:
                    self.logger.warning(f"Error comparing config in {exp_dir}: {e}")
        
        return matching_experiments
    
    def _configs_match(self, config1, config2):
        """Compare two configs for equality."""
        if set(config1.keys()) != set(config2.keys()):
            return False
            
        for key in config1.keys():
            if config1[key] != config2[key]:
                return False
                    
        return True
    
    def _find_latest_checkpoint(self, directory):
        """Find the latest checkpoint in a directory."""
        best_model_path = os.path.join(directory, "best_model.pt")
        if os.path.exists(best_model_path):
            return best_model_path
            
        latest_checkpoint = None
        latest_epoch = -1
        
        for filename in os.listdir(directory):
            if filename.startswith("checkpoint_epoch_"):
                try:
                    epoch_num = int(filename.split("_")[-1].split(".")[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint = os.path.join(directory, filename)
                except (ValueError, IndexError):
                    continue
                    
        return latest_checkpoint
    
    def train(
        self,
        encoder,
        generator,
        dataloader,
        optimizer,
        loss_manager,
        scheduler=None,
        device=None,
        output_dir='./outputs',
        config=None,
    ):
        """Train the model with W&B logging."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder.to(device)
        generator.model.to(device)
        
        stats = {
            'train_losses': [],
            'eval_losses': [],
            'best_epoch': 0,
            'total_time': 0,
        }
        
        # If config is provided, use hash-based output directory
        if config is not None:
            output_dir = get_output_dir(config, base_dir=output_dir)
            # Log the used output directory
            self.logger.info(f"Using hash-based output directory: {output_dir}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Using specified output directory: {output_dir}")
        
        # Check for existing checkpoints in the output directory
        best_model_path = os.path.join(output_dir, "best_model.pt")
        last_checkpoint = None
        start_epoch = 0
        
        # Try to find the latest checkpoint if it exists
        for filename in os.listdir(output_dir):
            if filename.startswith("checkpoint_epoch_"):
                try:
                    epoch_num = int(filename.split("_")[-1].split(".")[0])
                    if last_checkpoint is None or epoch_num > start_epoch:
                        last_checkpoint = os.path.join(output_dir, filename)
                        start_epoch = epoch_num
                except (ValueError, IndexError):
                    continue
        if start_epoch == 0 and best_model_path is not None and os.path.exists(best_model_path):
            last_checkpoint = best_model_path
        
        # If no checkpoint found and we have a config, try to find a similar experiment
        if last_checkpoint is None and config is not None:
            self.logger.info("No checkpoints found. Looking for similar experiments with different epoch counts...")
            
            # Get experiment name from directory
            dir_name = os.path.basename(output_dir)
            experiment_name = dir_name.split('_')[0]  # Assume format is "name_hash"
            
            # Find base directory (parent of output_dir)
            base_dir = os.path.dirname(output_dir)
            
            # Find similar experiments with same name but different epoch counts
            similar_experiments = self._find_similar_experiment_by_name(experiment_name, config, base_dir)
            
            print("similar_experiments", similar_experiments)
            if similar_experiments:
                # Sort by epoch count (descending) to prioritize those with more epochs
                similar_experiments.sort(key=lambda x: x[1], reverse=True)
                
                for exp_dir, num_epochs in similar_experiments:
                    self.logger.info(f"Checking for checkpoints in similar experiment with {num_epochs} epochs: {exp_dir}")
                    
                    # Find the latest checkpoint in this experiment
                    checkpoint_path = self._find_latest_checkpoint(exp_dir)
                    
                    if checkpoint_path:
                        # Copy the checkpoint to our current directory
                        new_checkpoint_path = os.path.join(output_dir, os.path.basename(checkpoint_path))
                        try:
                            shutil.copy2(checkpoint_path, new_checkpoint_path)
                            self.logger.info(f"Copied checkpoint from similar experiment: {checkpoint_path} -> {new_checkpoint_path}")
                            last_checkpoint = new_checkpoint_path
                            break
                        except Exception as e:
                            self.logger.error(f"Error copying checkpoint: {e}")
        
        # Resume from checkpoint if found
        step = 0
        if last_checkpoint is not None and os.path.exists(last_checkpoint):
            self.logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            checkpoint = torch.load(last_checkpoint, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            generator.model.load_state_dict(checkpoint['generator_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            if 'step' in checkpoint:
                step = checkpoint['step'] + 1
            else:
                step = (start_epoch - 1) * len(dataloader) + 1
            # Log resuming to W&B
            if wandb.run is not None:
                wandb.run.summary["resumed_from_epoch"] = start_epoch
        
        start_time = time.time()
        self.logger.info(f"Starting training on {device}...")
        
        # Main training loop
        for epoch in range(start_epoch, self.num_epochs):
            encoder.train()
            generator.model.train()
            
            epoch_losses = []
            
            # Create progress bar if requested
            if self.use_tqdm:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            else:
                pbar = dataloader
            
            # Train for one epoch
            for batch_idx, batch in enumerate(pbar):
                # Handle samples which can be either a tensor or a dictionary
                loss, losses = loss_manager.loss(encoder, generator, batch, device)
                
                optimizer.zero_grad()
                
                # Backpropagate
                loss.backward()
                optimizer.step()
                
                # Record loss
                epoch_losses.append(loss.item())
                
                # Log every log_interval batches
                if batch_idx % self.log_interval == 0:
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
                    if self.use_tqdm:
                        pbar.set_postfix(loss=f"{loss.item():.6f}")
                    
                    # Log batch metrics to W&B
                    if wandb.run is not None:
                        wandb.log({
                            "batch/loss": loss.item(),
                            "batch/step": step,
                            "batch/epoch": epoch + 1,
                        } | {f'batch/{k}': v for k, v in losses.items()}, step=step)

                if self.sub_epoch and (step % self.sub_epoch_interval == 0) and (step != 0):
                    sub_epoch = step // self.sub_epoch_interval
                    if scheduler is not None:
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
                        self.logger.info(f"Learning rate: {current_lr:.6f}")
                    if (sub_epoch + 1) % self.save_interval == 0:
                        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'encoder_state_dict': encoder.state_dict(),
                            'generator_state_dict': generator.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'step': step,
                        }, checkpoint_path)
                        self.logger.info(f"Saved checkpoint to {checkpoint_path}")      

                step += 1
            
            # Calculate average loss for this epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            stats['train_losses'].append(avg_epoch_loss)
            
            # Step the scheduler if provided
            if scheduler is not None and not self.sub_epoch:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                self.logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.6f}")
            
            # Log epoch metrics to W&B
            if wandb.run is not None:
                wandb_log = {
                    "epoch/train_loss": avg_epoch_loss,
                    "epoch/epoch": epoch + 1,
                }
                
                if scheduler is not None:
                    wandb_log["epoch/learning_rate"] = scheduler.get_last_lr()[0]
                
                wandb.log(wandb_log, step=step)
            
            # Save model checkpoint at regular intervals
            if (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_epoch_loss,
                    'step': step,
                }, checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Log model checkpoint to W&B
                if wandb.run is not None and (epoch + 1) == self.num_epochs:
                    wandb.save(checkpoint_path)
            
            # Evaluation and early stopping logic
            if ((epoch + 1) % self.eval_interval == 0 or (epoch + 1) == self.num_epochs):#  and False:
                eval_loss = self._evaluate(encoder, generator, dataloader, device)
                stats['eval_losses'].append(eval_loss)
                
                self.logger.info(f"Evaluation Loss: {eval_loss:.6f}")
                
                # Log evaluation metrics to W&B
                if wandb.run is not None:
                    wandb.log({
                        "epoch/eval_loss": eval_loss,
                        "epoch/epoch": epoch + 1,
                    }, step=step)

                # Generate some samples with the trained model
                samples = self.generate_samples(encoder, generator, dataloader)
                if samples is not None:
                    self.logger.info(f"Original samples shape: {samples.get('original', 'N/A').shape if hasattr(samples.get('original', None), 'shape') else 'N/A'}")
                    self.logger.info(f"Generated samples shape: {samples.get('generated', 'N/A').shape if hasattr(samples.get('generated', None), 'shape') else 'N/A'}")
                
                    # Log generated samples to W&B (optional)
                    if wandb.run is not None:
                        n_examples = min(6, len(samples['original']))
                        # Handle different types of samples
                        if 'generated_texts' in samples:
                            n_examples_per_example = min(6, len(samples['original_texts'][0]))
                            # For text data, use our text visualization
                            text_output_dir = os.path.join(output_dir, f"text_samples_epoch_{epoch+1}")
                            visualize_text_data(
                                text_output_dir,
                                samples['original_texts'],
                                samples['generated_texts'],
                            )

                            # Create a dataframe with original and generated texts, first need to flatten into sets
                            flat_original = []
                            flat_generated = []
                            set_indices = []
                            for i in range(n_examples):
                                for j in range(n_examples_per_example):
                                    flat_original.append(samples['original_texts'][i][j])
                                    flat_generated.append(samples['generated_texts'][i][j])
                                    set_indices.append(i)

                            import pandas as pd
                            df = pd.DataFrame({
                                'original': flat_original,
                                'generated': flat_generated,
                                'set_index': set_indices
                            })

                            print(df.head())

                            # Log a single table with both original and generated texts
                            wandb.log({
                                "epoch/text_samples": wandb.Table(dataframe=df)
                            }, step=step)
                            
                        elif 'original' in samples and 'generated' in samples and hasattr(samples['original'], 'shape'):
                            # For numerical or image data, use the original visualization
                            # We'll log just a few samples to avoid excessive data transfer
                            n_examples = min(6, samples['original'].shape[0])
                            for i in range(n_examples):
                                save_path = os.path.join(output_dir, f"pairplot_{i}_epoch_{epoch+1}.png")
                                visualize_data(
                                    save_path, samples['original'][i], samples['generated'][i]
                                )
                                wandb.log({
                                    f"samples/generated_{i}": wandb.Image(save_path)
                                })
                
                # Check if this is the best model so far
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    stats['best_epoch'] = epoch + 1
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'encoder_state_dict': encoder.state_dict(),
                        'generator_state_dict': generator.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': eval_loss,
                        'step': step,
                    }, best_model_path)
                    self.logger.info(f"New best model saved to {best_model_path}")
                    
                    # Log best model to W&B
                    if wandb.run is not None:
                        wandb.run.summary["best_epoch"] = epoch + 1
                        wandb.run.summary["best_loss"] = eval_loss
                        wandb.save(best_model_path)
                    
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    self.logger.info(f"No improvement for {self.no_improve_count} evaluations")
                
                # Early stopping check
                if self.early_stopping and self.no_improve_count >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    
                    # Log early stopping to W&B
                    if wandb.run is not None:
                        wandb.run.summary["stopped_early"] = True
                        wandb.run.summary["stopped_epoch"] = epoch + 1
                    
                    break
        
        # Record total training time
        stats['total_time'] = time.time() - start_time
        self.logger.info(f"Training completed in {stats['total_time']:.2f} seconds")
        
        # Log final stats to W&B
        if wandb.run is not None:
            wandb.run.summary["total_time"] = stats['total_time']
            if stats['train_losses']:
                wandb.run.summary["final_train_loss"] = stats['train_losses'][-1]
            if stats['eval_losses']:
                wandb.run.summary["final_eval_loss"] = stats['eval_losses'][-1]
        
        return output_dir, stats
    
    def _evaluate(self, encoder, generator, dataloader, device, num_eval_batches=10):
        """Run evaluation and return average loss."""
        encoder.eval()
        generator.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if num_eval_batches is not None and num_batches >= num_eval_batches:
                    break
                # Handle samples which can be either a tensor or a dictionary
                if isinstance(batch['samples'], torch.Tensor):
                    samples = batch['samples'].to(device)
                    latent = encoder(samples)
                    loss = generator.loss(samples.view(-1, *samples.shape[2:]), latent)
                else:
                    # For dictionary samples (like PubMed dataset), move tensors to device
                    samples = {}
                    for key, value in batch['samples'].items():
                        if isinstance(value, torch.Tensor):
                            samples[key] = value.to(device)
                        else:
                            samples[key] = value
                    
                    # Encode samples to latent space
                    latent = encoder(samples)
                    
                    # Calculate loss
                    loss = generator.loss(samples, latent)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def generate_samples(self, encoder, generator, dataloader, num_samples=None, device=None):
        """Generate samples using the trained model."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        encoder.to(device)
        generator.model.to(device)
        
        encoder.eval()
        generator.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle samples which can be either a tensor or a dictionary
                if isinstance(batch['samples'], torch.Tensor):
                    samples = batch['samples'].to(device)
                    batch_size, set_size, *data_shape = samples.shape
                    if num_samples is None:
                        num_samples = set_size
                    
                    # Encode samples to latent space
                    latent = encoder(samples)
                    
                    # Generate new samples
                    generated = generator.sample(latent, num_samples=num_samples)
                    
                    return {
                        'original': samples.cpu(),
                        'generated': generated.cpu()
                    }
                else:
                    # For dictionary samples (like PubMed dataset), move tensors to device
                    samples = {}
                    for key, value in batch['samples'].items():
                        if isinstance(value, torch.Tensor):
                            samples[key] = value.to(device)
                        else:
                            samples[key] = value
                    
                    # Keep raw texts for reference
                    raw_texts = batch.get('raw_texts', None)
                    set_size = len(raw_texts)
                    num_sets = len(raw_texts[0])

                    # reshape raw_texts list from [set_size, num_samples] to [num_samples, set_size]
                    raw_texts = [[raw_texts[j][i] for j in range(set_size)] for i in range(num_sets)]

                    # Encode samples to latent space
                    latent = encoder(samples)
                    
                    # Generate new samples
                    generated = generator.sample(latent, num_samples=set_size, return_texts=True)
                    
                    if isinstance(generated, tuple):
                        # If generator returns both token ids and decoded texts
                        generated_ids, generated_texts = generated
                        return {
                            'original': samples,
                            'generated': generated_ids.cpu(),
                            'original_texts': raw_texts,
                            'generated_texts': generated_texts
                        }
                    else:
                        return {
                            'original': samples,
                            'generated': generated.cpu(),
                            'original_texts': raw_texts
                        }