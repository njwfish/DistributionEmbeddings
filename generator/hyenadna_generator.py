import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConditionedHyenaDNA(nn.Module):
    """HyenaDNA model conditioned on a latent vector representing a distribution of DNA sequences."""
    
    def __init__(
        self, 
        latent_dim=32,
        condition_dim=128,
        d_model=128,
        n_layer=6,
        vocab_size=11,  # A, C, G, T, N
        max_seq_len=512,
        condition_method="prefix"
    ):
        """
        Initialize a conditioned HyenaDNA model.
        
        Args:
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            d_model: Model dimension
            n_layer: Number of layers
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            condition_method: How to condition the model ('prefix' or 'additive')
        """
        super().__init__()
        
        # Initialize HyenaDNA model
        from generator.hyenadna import HyenaDNAModel
        
        # Adjust max_seq_len to account for prefix token if using prefix conditioning
        self.actual_max_seq_len = max_seq_len + 1 if condition_method == "prefix" else max_seq_len
        
        # Configure parameters for the HyenaDNA model
        # The layer parameter needs to be a dictionary with specific settings
        hyena_layer_config = {
            'd_model': d_model,
            'l_max': self.actual_max_seq_len,  # Use adjusted max length for HyenaOperator
            'order': 2,            # Default Hyena recurrence depth
            'filter_order': 64     # Default filter order
        }
        
        self.model = HyenaDNAModel(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=4 * d_model,
            vocab_size=vocab_size,
            max_position_embeddings=self.actual_max_seq_len,  # Use adjusted max length
            layer=hyena_layer_config,  # Pass the layer config as a dictionary
            attn_layer_idx=None,       # No attention layers
            attn_cfg=None,             # No attention config
            layer_norm_epsilon=1e-5,
            embed_dropout=0.1,
            resid_dropout=0.0,
            residual_in_fp32=False
        )
        
        # Add a language model head for next token prediction
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.condition_method = condition_method
        self.hidden_dim = d_model
        self.max_seq_len = max_seq_len  # Keep original max_seq_len for input validation
        self.vocab_size = vocab_size
        
        # Project latent to condition dimension
        self.condition_proj = nn.Sequential(
            nn.Linear(latent_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, self.hidden_dim)
        )
        
        if self.condition_method == "prefix":
            # No additional projection needed for prefix conditioning
            # We'll directly use the condition as a prefix embedding
            pass
        elif self.condition_method == "additive":
            # For additive conditioning
            self.condition_adapter = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU()
            )
        else:
            raise ValueError(f"Unknown conditioning method: {condition_method}")
    
    def forward(self, input_ids, attention_mask, latent):
        """
        Forward pass through the conditioned HyenaDNA model.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len] or [batch_size, set_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] or [batch_size, set_size, seq_len] 
            latent: Latent distribution embedding [batch_size, latent_dim]
        
        Returns:
            Logits for next token prediction
        """
        batch_size = input_ids.shape[0]
        
        # Validate sequence length
        if len(input_ids.shape) == 3:
            seq_len = input_ids.shape[-1]
        else:
            seq_len = input_ids.shape[1]
            
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum allowed length {self.max_seq_len}")
        
        # Project latent to correct dimension
        condition = self.condition_proj(latent)
        
        if self.condition_method == "prefix":
            # Handle set_size dimension if present
            if len(input_ids.shape) == 3:  # [batch_size, set_size, seq_len]
                batch_size, set_size, seq_len = input_ids.shape
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Get input embeddings from the model
            input_embeds = self.model.backbone.embeddings.word_embeddings(input_ids)
            
            # Create prefix embedding from condition
            prefix_embeds = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Concatenate prefix with input embeddings
            combined_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
            
            # Generate position IDs for the combined sequence
            combined_seq_length = combined_embeds.size(1)
            position_ids = torch.arange(0, combined_seq_length, dtype=torch.long, device=combined_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Forward pass through the model using inputs_embeds
            hidden_states = self.model.backbone(
                position_ids=position_ids,
                inputs_embeds=combined_embeds
            )
            
            # Get logits from language model head
            logits = self.lm_head(hidden_states)
            
            # Remove prefix logits
            logits = logits[:, 1:, :]  # Remove the first position (prefix)
            
        elif self.condition_method == "additive":
            # Handle set_size dimension if present
            if len(input_ids.shape) == 3:  # [batch_size, set_size, seq_len]
                set_size, seq_len = input_ids.shape[1:]
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Process with the backbone first
            hidden_states = self.model.backbone(input_ids=input_ids)
            
            # Process condition
            processed_condition = self.condition_adapter(condition).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Broadcast processed_condition to all positions in the sequence
            processed_condition = processed_condition.expand(-1, hidden_states.size(1), -1)
            
            # Add condition to hidden states
            hidden_states = hidden_states + processed_condition
            
            # Project to vocabulary with language model head
            logits = self.lm_head(hidden_states)
        
        return logits


class HyenaDNAGenerator:
    """Generator class using conditioned HyenaDNA for the distribution embeddings framework."""
    
    def __init__(
        self,
        latent_dim=32,
        condition_dim=128,
        d_model=128,
        n_layer=6,
        vocab_size=None,  # Will be determined from the tokenizer
        max_seq_len=512,
        condition_method="prefix",
        temperature=1.0,
    ):
        """
        Initialize the HyenaDNA generator.
        
        Args:
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            d_model: Model dimension
            n_layer: Number of layers
            vocab_size: If provided, sets the vocabulary size. Otherwise determined from tokenizer.
            max_seq_len: Maximum sequence length
            condition_method: How to condition the model ('prefix' or 'additive')
            temperature: Sampling temperature
        """
        # Initialize tokenizer first so we can determine the vocab size
        from generator.hyenadna import CharacterTokenizer
        dna_vocab = ["A", "C", "G", "T", "N"]
        self.tokenizer = CharacterTokenizer(characters=dna_vocab, model_max_length=max_seq_len)
        
        # If vocab_size is None, determine it from the tokenizer
        if vocab_size is None:
            # The tokenizer adds 7 special tokens plus the actual vocabulary
            # [CLS], [SEP], [BOS], [MASK], [PAD], [RESERVED], [UNK]
            vocab_size = len(dna_vocab) + 7
        
        # Now initialize the model with the correct vocabulary size
        self.model = ConditionedHyenaDNA(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            condition_method=condition_method
        )
        
        self.temperature = temperature
        self.max_seq_len = max_seq_len
    
    def loss(self, x, latent):
        """
        Calculate the loss for the generator.
        
        Args:
            x: Dictionary containing 'hyena_input_ids' and 'hyena_attention_mask'
            latent: Latent distribution embedding
        
        Returns:
            Negative log likelihood loss
        """
        input_ids = x['hyena_input_ids']
        attention_mask = x['hyena_attention_mask']
        
        # Debug shape
        batch_size, *rest_dims = input_ids.shape

        # Handle set_size dimension consistently if present
        if len(input_ids.shape) == 3:  # [batch_size, set_size, seq_len]
            batch_size, set_size, seq_len = input_ids.shape
            
            # Flatten batch and set dimensions for both inputs
            # This must match how forward() handles the dimensions
            flat_input_ids = input_ids.reshape(batch_size * set_size, seq_len)
            flat_attention_mask = attention_mask.reshape(batch_size * set_size, seq_len)
            
            # Repeat latent for each element in the set
            flat_latent = latent.unsqueeze(1).repeat(1, set_size, 1).reshape(batch_size * set_size, -1)
        else:
            # No set dimension, use as is
            flat_input_ids = input_ids
            flat_attention_mask = attention_mask
            flat_latent = latent
        
        # Get model outputs
        logits = self.model(flat_input_ids, flat_attention_mask, flat_latent)

        # For causal language modeling, shift the targets
        shift_logits = logits[:, :-1, :]  # Remove last position (shape: [batch*set, seq_len-1, vocab])
        shift_labels = flat_input_ids[:, 1:]   # Remove first position (shape: [batch*set, seq_len-1])
        shift_attention_mask = flat_attention_mask[:, 1:]  # Remove first position
        

        # Flatten the tensors to prepare for loss calculation
        # shift_logits: [batch*set, seq_len-1, vocab] -> [(batch*set)*(seq_len-1), vocab]
        # shift_labels: [batch*set, seq_len-1] -> [(batch*set)*(seq_len-1)]
        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)
        flat_attention = shift_attention_mask.reshape(-1)
        

        # Only compute loss on tokens that are valid (attention mask = 1)
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        active_loss = flat_attention == 1
        active_logits = flat_logits[active_loss]
        active_labels = flat_labels[active_loss]

        loss = loss_fct(active_logits, active_labels)
        
        return loss
    
    def _generate_sequences(self, input_ids, attention_mask, latent, max_length):
        """
        Generate sequences using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            latent: Latent distribution embedding
            max_length: Maximum sequence length
        
        Returns:
            Generated sequences
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        cur_len = input_ids.size(1)
        
        # Keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Get the actual EOS token ID 
        eos_token_id = self.tokenizer.eos_token_id  # [SEP] token with ID=1
        
        while cur_len < max_length:
            # Get next-token logits
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, latent)[:, -1, :]
            
            # Apply temperature
            logits = logits / self.temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Append next tokens to sequences
            next_tokens = next_tokens * unfinished_sequences + eos_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)
            
            # Update which sequences are finished
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).long())
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
                
            cur_len += 1
        
        return input_ids
    
    def sample(self, latent, num_samples=1, return_texts=False):
        """
        Sample DNA sequences from the generator.
        
        Args:
            latent: Latent distribution embedding
            num_samples: Number of samples to generate per latent
            return_texts: Whether to also return decoded texts
        
        Returns:
            Generated token IDs, and optionally decoded texts
        """
        self.model.eval()
        
        batch_size = latent.size(0)
        device = latent.device
        
        # Prepare for generation
        start_tokens = torch.full(
            (batch_size, 1),
            self.tokenizer.cls_token_id,  # Use BOS token ID = 2
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.ones_like(start_tokens)
        
        # Prepare to collect results
        all_generated_ids = []
        
        for _ in range(num_samples):
            # Generate DNA sequences
            with torch.no_grad():
                # Use a copy of the latent for each sample
                generated_ids = self._generate_sequences(
                    input_ids=start_tokens.clone(),
                    attention_mask=attention_mask.clone(),
                    latent=latent,
                    max_length=self.max_seq_len
                )
            
            all_generated_ids.append(generated_ids)
        
        # Make sure all tensors have the same length by padding to the maximum length
        max_len = max(ids.size(1) for ids in all_generated_ids)
        padded_ids = []
        for ids in all_generated_ids:
            if ids.size(1) < max_len:
                # Pad with EOS tokens
                padding = torch.full(
                    (batch_size, max_len - ids.size(1)),
                    self.tokenizer.sep_token_id,
                    dtype=ids.dtype,
                    device=device
                )
                ids = torch.cat([ids, padding], dim=1)
            padded_ids.append(ids)
        
        # Combine samples
        result = torch.stack(padded_ids, dim=1)  # [batch_size, num_samples, seq_len]
        
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
            print("text shape", len(all_texts), len(all_texts[0]), len(all_texts[0][0]))
            
            return result, all_texts
        
        return result