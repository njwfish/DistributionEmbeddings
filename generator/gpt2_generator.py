import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
import numpy as np

class ConditionedGPT2(nn.Module):
    """GPT-2 model conditioned on a latent vector representing a document distribution."""
    
    def __init__(
        self, 
        gpt2_model_name="gpt2", 
        latent_dim=32,
        condition_dim=768,
        freeze_gpt2=False,
        condition_method="prefix"
    ):
        """
        Initialize a conditioned GPT-2 model.
        
        Args:
            gpt2_model_name: Name of the pretrained GPT-2 model
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            freeze_gpt2: Whether to freeze the GPT-2 parameters
            condition_method: How to condition GPT-2 ('prefix', 'additive', or 'concat')
        """
        super().__init__()
        
        # Initialize GPT-2 model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        
        # Freeze GPT-2 if specified
        if freeze_gpt2:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        self.condition_method = condition_method
        
        # Project latent to correct dimension
        if self.condition_method == "prefix":
            # For prefix conditioning, project to hidden states
            self.hidden_dim = self.gpt2.config.hidden_size
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
            )
        elif self.condition_method == "additive":
            # For additive conditioning, project to hidden states
            self.hidden_dim = self.gpt2.config.hidden_size
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
            )
        elif self.condition_method == "concat":
            # For concat conditioning, we'll concat the latent to the embedding
            self.embed_dim = self.gpt2.config.n_embd
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.embed_dim)
            )
            # Create a new embedding that includes the condition
            self.new_embeddings = nn.Linear(
                self.embed_dim * 2,  # Original embedding + condition
                self.embed_dim
            )
        else:
            raise ValueError(f"Unknown conditioning method: {condition_method}")
    
    def forward(self, input_ids, attention_mask, latent):
        """
        Forward pass through the conditioned GPT-2 model.
        
        Args:
            input_ids: Tensor of token IDs
            attention_mask: Attention mask
            latent: Latent distribution embedding
        
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
                # For PubMed dataset, reshape if needed
                set_size, seq_len = attention_mask.shape[1:]
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            prefix_attention = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
            
            # Get GPT-2 embeddings for the input sequence
            inputs_embeds = self.gpt2.transformer.wte(input_ids)
            
            # Create prefix embedding
            prefix_embeds = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Concatenate prefix embedding with input embeddings
            embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            
            # Run through GPT-2 with custom embeddings
            outputs = self.gpt2(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Get logits and remove the prefix logit
            logits = outputs.logits[:, 1:, :]
            
        elif self.condition_method == "additive":
            # Add the condition to each token embedding
            # Handle attention mask dimension properly - check shape and reshape if needed
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # For PubMed dataset, reshape if needed
                set_size, seq_len = attention_mask.shape[1:]
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            outputs = self.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Get the hidden states and add the condition
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            
            # Add the condition to each position
            condition = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            hidden_states = hidden_states + condition
            
            # Project back to vocabulary
            logits = self.gpt2.lm_head(hidden_states)
            
        elif self.condition_method == "concat":
            # Concatenate the condition to each token embedding
            # Handle attention mask dimension properly - check shape and reshape if needed
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # For PubMed dataset, reshape if needed
                set_size, seq_len = attention_mask.shape[1:]
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Get token embeddings
            token_embeds = self.gpt2.transformer.wte(input_ids)  # [batch_size, seq_len, embed_dim]
            
            # Expand condition to match sequence length
            seq_len = input_ids.shape[1]
            expanded_condition = condition.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embed_dim]
            
            # Concatenate along embedding dimension
            combined_embeds = torch.cat([token_embeds, expanded_condition], dim=2)  # [batch_size, seq_len, embed_dim*2]
            
            # Project back to original embedding size
            new_embeds = self.new_embeddings(combined_embeds)
            
            # Run through GPT-2 with custom embeddings
            outputs = self.gpt2(
                inputs_embeds=new_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            logits = outputs.logits
        
        return logits

class GPT2Generator:
    """Generator class using conditioned GPT-2 for the distribution embeddings framework."""
    
    def __init__(
        self,
        gpt2_model_name="gpt2",
        latent_dim=32,
        condition_dim=768,
        freeze_gpt2=False,
        condition_method="prefix",
        temperature=1.0,
        top_p=0.9,
        max_length=128
    ):
        """
        Initialize the GPT-2 generator.
        
        Args:
            gpt2_model_name: Name of the pretrained GPT-2 model
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            freeze_gpt2: Whether to freeze the GPT-2 parameters
            condition_method: How to condition GPT-2
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_length: Maximum length for generation
        """
        self.model = ConditionedGPT2(
            gpt2_model_name=gpt2_model_name,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            freeze_gpt2=freeze_gpt2,
            condition_method=condition_method
        )
        
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        
        # Initialize tokenizer (for generation)
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def loss(self, x, latent):
        """
        Calculate the loss for the generator.
        
        Args:
            x: Dictionary containing 'gpt2_input_ids' and 'gpt2_attention_mask'
            latent: Latent distribution embedding
        
        Returns:
            Negative log likelihood loss
        """
        input_ids = x['gpt2_input_ids']
        attention_mask = x['gpt2_attention_mask']
        
        # Shift for causal language modeling: predict each token using previous tokens
        shift_logits = self.model(input_ids, attention_mask, latent)[:, :-1, :]
        
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
        Sample text from the generator.
        
        Args:
            latent: Latent distribution embedding
            num_samples: Number of samples to generate per latent
            return_texts: Whether to also return decoded texts
        
        Returns:
            Generated token IDs, and optionally decoded texts
        """
        device = latent.device
        batch_size = latent.shape[0]
        
        # Prepare for generation
        start_tokens = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size, device=device)
        attention_mask = torch.ones_like(start_tokens)
        
        # Prepare to collect results
        all_generated_ids = []
        
        for _ in range(num_samples):
            # Use model for text generation
            with torch.no_grad():
                generated_ids = self._generate_text(
                    input_ids=start_tokens,
                    attention_mask=attention_mask,
                    latent=latent,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p
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
                    self.tokenizer.eos_token_id,
                    dtype=ids.dtype,
                    device=device
                )
                ids = torch.cat([ids, padding], dim=1)
            padded_ids.append(ids)
        
        # Combine samples
        result = torch.stack(padded_ids, dim=1)  # [batch_size, num_samples, seq_len]
        print(result.shape)
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
                        text = "Generated text was empty."
                    batch_texts.append(text)
                all_texts.append(batch_texts)
            print(len(all_texts), len(all_texts[0]))
            
            return result, all_texts
        
        return result
    
    def _generate_text(self, input_ids, attention_mask, latent, max_length, temperature, top_p):
        """Helper method for text generation using the conditioned GPT-2."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(batch_size).fill_(1)
        
        # Initialize with the starting tokens
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        
        # Ensure we have the right shape for attention mask
        if len(cur_attention_mask.shape) > 2:
            # Reshape to 2D for generation
            cur_attention_mask = cur_attention_mask.squeeze(0)
        
        for _ in range(max_length - 1):  # -1 because we start with one token
            # Get logits for the next token
            with torch.no_grad():
                logits = self.model(cur_input_ids, cur_attention_mask, latent)
            
            # Focus on the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-p nucleus sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = 0
                
                # Shift the indices to the right to prevent removing the first token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a sparse mask
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Add the sampled token to the input
            tokens = torch.cat([cur_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update the attention mask
            attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((batch_size, 1), device=device)], dim=1
            )
            
            # Update which sequences are finished
            eos_token_id = self.tokenizer.eos_token_id
            next_tokens = next_tokens.masked_fill(~unfinished_sequences.bool(), eos_token_id)
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id)
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
            
            # Update input for the next iteration
            cur_input_ids = tokens
            cur_attention_mask = attention_mask
        
        # Return the final tokens
        return cur_input_ids 