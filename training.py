import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import logging
import wandb
from utils.hash_utils import get_output_dir, find_matching_output_dir
from utils.visualization import visualize_data, visualize_text_data

class Trainer:
    def __init__(
        self,
        num_epochs=100,
        log_interval=10,
        save_interval=20,
        eval_interval=5,
        early_stopping=True,
        patience=10,
        use_tqdm=True,
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
        self.early_stopping = early_stopping
        self.patience = patience
        self.use_tqdm = use_tqdm
        
        self.logger = logging.getLogger(__name__)
        self.best_loss = float('inf')
        self.no_improve_count = 0
        
    def train(
        self,
        encoder,
        generator,
        dataloader,
        optimizer,
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
        
        # Resume from checkpoint if found
        if last_checkpoint is not None and os.path.exists(last_checkpoint):
            self.logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            checkpoint = torch.load(last_checkpoint, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            generator.model.load_state_dict(checkpoint['generator_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            
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
                if isinstance(batch['samples'], torch.Tensor):
                    samples = batch['samples'].to(device)
                    latent = encoder(samples)  # latent is num samples x num sets x latent dim
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
                        step = epoch * len(dataloader) + batch_idx
                        wandb.log({
                            "batch/loss": loss.item(),
                            "batch/step": step,
                            "batch/epoch": epoch + 1,
                        }, step=step)
            
            # Calculate average loss for this epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            stats['train_losses'].append(avg_epoch_loss)
            
            # Step the scheduler if provided
            if scheduler is not None:
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
                
                wandb.log(wandb_log, step=(epoch + 1) * len(dataloader))
            
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
                }, checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Log model checkpoint to W&B
                if wandb.run is not None and (epoch + 1) == self.num_epochs:
                    wandb.save(checkpoint_path)
            
            # Evaluation and early stopping logic
            if (epoch + 1) % self.eval_interval == 0 or (epoch + 1) == self.num_epochs:
                eval_loss = self._evaluate(encoder, generator, dataloader, device)
                stats['eval_losses'].append(eval_loss)
                
                self.logger.info(f"Evaluation Loss: {eval_loss:.6f}")
                
                # Log evaluation metrics to W&B
                if wandb.run is not None:
                    wandb.log({
                        "epoch/eval_loss": eval_loss,
                        "epoch/epoch": epoch + 1,
                    }, step=(epoch + 1) * len(dataloader))

                # Generate some samples with the trained model
                samples = self.generate_samples(encoder, generator, dataloader)
                if samples is not None:
                    self.logger.info(f"Original samples shape: {samples.get('original', 'N/A').shape if hasattr(samples.get('original', None), 'shape') else 'N/A'}")
                    self.logger.info(f"Generated samples shape: {samples.get('generated', 'N/A').shape if hasattr(samples.get('generated', None), 'shape') else 'N/A'}")
                
                    # Log generated samples to W&B (optional)
                    if wandb.run is not None:
                        # Handle different types of samples
                        if 'generated_texts' in samples:
                            # For text data, use our text visualization
                            text_output_dir = os.path.join(output_dir, f"text_samples_epoch_{epoch+1}")
                            visualize_text_data(
                                text_output_dir,
                                samples['original_texts'],
                                samples['generated_texts']
                            )

                            # Create a combined table with original and generated texts side by side
                            paired_data = []
                            for i in range(len(samples['original_texts'])):
                                for j in range(len(samples['original_texts'][i])):
                                    paired_data.append([samples['original_texts'][i][j], samples['generated_texts'][i][j]])
                            
                            # Log a single table with both original and generated texts
                            wandb.log({
                                "epoch/text_samples": wandb.Table(data=paired_data, columns=["original", "generated"])
                            }, step=(epoch + 1) * len(dataloader))
                            
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
                        'loss': eval_loss,
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
    
    def _evaluate(self, encoder, generator, dataloader, device):
        """Run evaluation and return average loss."""
        encoder.eval()
        generator.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
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
    
    def generate_samples(self, encoder, generator, dataloader, num_samples=100, device=None):
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
                    num_sets = len(raw_texts)
                    set_size = len(raw_texts[0])
                    # reshape raw_texts list from [set_size, num_samples] to [num_samples, set_size]
                    raw_texts = [[raw_texts[j][i] for j in range(num_sets)] for i in range(set_size)]

                    # Encode samples to latent space
                    latent = encoder(samples)
                    
                    # Generate new samples
                    generated = generator.sample(latent, num_samples=num_sets, return_texts=True)
                    
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
            