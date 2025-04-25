import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# This module has been replaced by a more complex sequential network in ConditionedProgen2
# class ZToPrefix(nn.Module):
#     def __init__(self, input_dim, prefix_length, d_model):
#         super().__init__()
#         # A simple linear projection that outputs prefix_length * d_model values
#         self.fc = nn.Linear(input_dim, prefix_length * d_model)
#         self.prefix_length = prefix_length
#         self.d_model = d_model
#         
#     def forward(self, x):
#         # x shape: (batch_size, input_dim)
#         batch_size = x.size(0)
#         prefix = self.fc(x)            # Shape: (batch_size, prefix_length * d_model)
#         prefix = prefix.view(batch_size, self.prefix_length, self.d_model)  # Reshape to (batch_size, prefix_length, d_model)
#         return prefix

class ConditionedProgen2(nn.Module):
    """Progen2 model conditioned on a latent vector representing a distribution of protein sequences."""
    
    def __init__(
        self, 
        progen2_name='hugohrban/progen2-medium', 
        latent_dim=32,
        condition_dim=256,
        freeze_progen2=False,
        condition_method="prefix"
    ):
        """
        Initialize a conditioned Progen2 model.
        
        Args:
            progen2_name: Name of the pretrained Progen2 model
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            freeze_progen2: Whether to freeze the Progen2 parameters
            condition_method: How to condition the model ('prefix' or 'additive')
        """
        super().__init__()
        
        # Initialize Progen2 model
        self.progen2 = AutoModelForCausalLM.from_pretrained(progen2_name, trust_remote_code=True)
        
        # Freeze Progen2 if specified
        if freeze_progen2:
            for param in self.progen2.parameters():
                param.requires_grad = False
        
        # Get the embedding dimension from the model config
        # Different models might use different attribute names
        if hasattr(self.progen2.config, 'hidden_size'):
            self.hidden_dim = self.progen2.config.hidden_size
        elif hasattr(self.progen2.config, 'n_embd'):
            self.hidden_dim = self.progen2.config.n_embd
        elif hasattr(self.progen2.config, 'embed_dim'):
            self.hidden_dim = self.progen2.config.embed_dim
        elif hasattr(self.progen2.config, 'd_model'):
            self.hidden_dim = self.progen2.config.d_model
        else:
            # Default value if none of the above attributes exist
            self.hidden_dim = 768
            print(f"Warning: Could not determine hidden dimension from model config. Using default: {self.hidden_dim}")
        
        self.condition_method = condition_method
        
        # Project latent to correct dimension (same approach as in GPT-2)
        if self.condition_method == "prefix":
            # For prefix conditioning, project to hidden states
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
            )
        elif self.condition_method == "additive":
            # For additive conditioning, project to hidden states
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
            )
        else:
            raise ValueError(f"Unknown conditioning method: {condition_method}")

    def forward(self, input_ids, attention_mask, latent):
        """
        Forward pass through the conditioned Progen2 model.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            latent: Latent distribution embedding [batch_size, latent_dim]
            
        Returns:
            Logits for next token prediction
        """
        batch_size = input_ids.shape[0]
        
        # Project latent to correct dimension
        condition = self.condition_proj(latent)
        
        if self.condition_method == "prefix":
            # Use the condition as a prefix hidden state
            # Create a prefix token
            
            # Handle attention mask dimension properly - check shape and reshape if needed
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # Reshape if needed
                set_size, seq_len = attention_mask.shape[1:]
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Add prefix to attention mask
            prefix_attention = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
            
            # Get embeddings for the input sequence (directly from token embeddings or wte)
            if hasattr(self.progen2.transformer, 'wte'):
                # Usually Progen2 uses wte for word token embeddings
                token_embeds = self.progen2.transformer.wte(input_ids)
            else:
                # Fallback to manual embedding lookup
                token_embeds = self.progen2.get_input_embeddings()(input_ids)
            
            # Create prefix embedding
            prefix_embeds = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Concatenate prefix embedding with input embeddings
            combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            
            # Run through Progen2 with custom embeddings
            outputs = self.progen2(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            
            # Get logits and remove the prefix logit
            logits = outputs.logits[:, 1:, :]
            
        elif self.condition_method == "additive":
            # Handle attention mask dimension properly - check shape and reshape if needed
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # Reshape if needed
                set_size, seq_len = attention_mask.shape[1:]
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Process with the model first
            outputs = self.progen2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Get the final hidden states
            hidden_states = outputs.hidden_states[-1]
            
            # Add the condition to each position
            condition = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            hidden_states = hidden_states + condition
            
            # Project back to vocabulary
            logits = self.progen2.lm_head(hidden_states)
        
        return logits


class Progen2Generator:
    """Generator class using conditioned Progen2 for the distribution embeddings framework."""
    
    def __init__(
        self,
        progen2_name="hugohrban/progen2-medium",
        latent_dim=32,
        condition_dim=256,
        freeze_progen2=False,
        condition_method="prefix",
        temperature=1.0,
        max_length=512
    ):
        """
        Initialize the Progen2 generator.
        
        Args:
            progen2_name: Name of the pretrained Progen2 model
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            freeze_progen2: Whether to freeze the Progen2 parameters
            condition_method: How to condition Progen2
            temperature: Sampling temperature
            max_length: Maximum length for generation
        """
        self.model = ConditionedProgen2(
            progen2_name=progen2_name,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            freeze_progen2=freeze_progen2,
            condition_method=condition_method
        )
        
        self.temperature = temperature
        self.max_length = max_length
        
        # Initialize tokenizer (for generation)
        self.tokenizer = AutoTokenizer.from_pretrained(progen2_name, trust_remote_code=True)
        
        # Add special tokens if they don't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<|pad|>'
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<|bos|>'
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '<|eos|>'
    
    def loss(self, x, latent):
        """
        Calculate the loss for the generator.
        
        Args:
            x: Dictionary containing 'progen_input_ids' and 'progen_attention_mask'
            latent: Latent distribution embedding
        
        Returns:
            Negative log likelihood loss
        """
        input_ids = x['progen_input_ids']
        attention_mask = x['progen_attention_mask']
        
        # Shift for causal language modeling: predict each token using previous tokens
        logits = self.model(input_ids, attention_mask, latent)
        shift_logits = logits[:, :-1, :]
        
        # Reshape input_ids if needed to match logits
        if len(input_ids.shape) == 3 and len(shift_logits.shape) == 3:
            if input_ids.shape[0] * input_ids.shape[1] == shift_logits.shape[0]:
                # Reshape input_ids to match the reshaped logits
                input_ids = input_ids.view(-1, input_ids.shape[-1])
        
        shift_labels = input_ids[:, 1:]
        
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        
        return loss

    def sample(self, latent, num_samples=1, return_texts=False):
        """
        Sample sequences from the conditioned model.
        
        Args:
            latent: Latent distribution embedding
            num_samples: Number of samples to generate per latent
            return_texts: Whether to also return decoded texts
        
        Returns:
            Generated token IDs, and optionally decoded texts
        """
        device = latent.device
        batch_size = latent.size(0)
        
        # Get BOS token ID for start of generation
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            bos_token_id = self.tokenizer.bos_token_id
        else:
            # Default to 1 if no BOS token is defined
            bos_token_id = 1
        
        # Initialize with the starting tokens
        start_ids = torch.tensor([[bos_token_id]] * batch_size, device=device)
        start_mask = torch.ones_like(start_ids)
        
        # Generate samples
        all_samples = []
        for _ in range(num_samples):
            # Add noise for diversity if generating multiple samples
            if num_samples > 1:
                noise_scale = 0.1
                noisy_latent = latent + noise_scale * torch.randn_like(latent)
            else:
                noisy_latent = latent
                
            with torch.no_grad():
                out = self._generate_text(
                    start_ids.clone(),
                    start_mask.clone(),
                    noisy_latent,
                    self.max_length,
                    self.temperature
                )
            all_samples.append(out)
        
        # Combine samples 
        result = torch.stack(all_samples, dim=1)  # [batch_size, num_samples, seq_len]
        
        if return_texts:
            # Decode texts
            all_texts = []
            for batch_idx in range(batch_size):
                batch_texts = []
                for sample_idx in range(num_samples):
                    ids = result[batch_idx, sample_idx]
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    # Make sure we have at least some text
                    if not text.strip():
                        text = "Generated sequence was empty."
                    batch_texts.append(text)
                all_texts.append(batch_texts)
            
            return result, all_texts
        
        return result

    def _generate_text(self, input_ids, attention_mask, latent, max_length, temperature=1.0):
        """
        Helper method for text generation using the conditioned Progen2 model.
        
        Args:
            input_ids: Starting token IDs
            attention_mask: Attention mask
            latent: Latent distribution embedding
            max_length: Maximum sequence length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize with the starting tokens
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        
        # Get EOS token ID for stopping generation
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens up to max_length or until all sequences have EOS
        for _ in range(max_length - cur_input_ids.size(1)):
            # Forward pass
            with torch.no_grad():
                logits = self.model(cur_input_ids, cur_attention_mask, latent)
            
            # Get logits for next token prediction (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # If a sequence is finished, use EOS token
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token
                )
            
            # Append next token to sequence
            cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
            
            # Update attention mask
            next_mask = torch.ones_like(next_token)
            cur_attention_mask = torch.cat([cur_attention_mask, next_mask], dim=1)
            
            # Mark sequences as finished if EOS token is generated
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                
                # If all sequences are finished, stop generation
                if finished.all():
                    break
        
        return cur_input_ids