import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class ZToPrefix(nn.Module):
    def __init__(self, input_dim, prefix_length, d_model):
        super().__init__()
        # A simple linear projection that outputs prefix_length * d_model values
        self.fc = nn.Linear(input_dim, prefix_length * d_model)
        self.prefix_length = prefix_length
        self.d_model = d_model
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        prefix = self.fc(x)            # Shape: (batch_size, prefix_length * d_model)
        prefix = prefix.view(batch_size, self.prefix_length, self.d_model)  # Reshape to (batch_size, prefix_length, d_model)
        return prefix

class ConditionedProgen2(nn.Module):
    def __init__(self, progen2_name='hugohrban/progen2-medium', 
                 latent_dim=128, prefix_length=10, ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(progen2_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(progen2_name, trust_remote_code=True)
        self.prefix_module = ZToPrefix(latent_dim, prefix_length, self.model.config.embed_dim)
        self.prefix_length = prefix_length

    def forward(self, input_ids, esm_embedding):
        # Get the prefix embeddings from the ESM embedding
        prefix_embeddings = self.prefix_module(esm_embedding)
        
        prefix_embeddings = prefix_embeddings.repeat(input_ids.shape[0]//prefix_embeddings.shape[0], 1, 1)

        input_embeddings = self.model.transformer.wte(input_ids)  # Get the token embeddings

        combined_embeddings = torch.cat((prefix_embeddings, input_embeddings), dim=1)
        
        # Forward pass through the model
        outputs = self.model(inputs_embeds=combined_embeddings)
        return outputs.logits[:, self.prefix_length:, :]
        
class Progen2Generator(nn.Module):
    def __init__(
        self,
        progen2_name="hugohrban/progen2-medium",
        latent_dim=128,
        prefix_length=10,
        temperature=1.0,
        max_length=512,
        device="cuda",
    ):
        super().__init__()
        self.model = ConditionedProgen2(
            progen2_name=progen2_name,
            latent_dim=latent_dim,
            prefix_length=prefix_length,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(progen2_name, trust_remote_code=True)
        self.temperature = temperature
        self.max_length = max_length
        self.device = device

    def loss(self, x, latent):
        input_ids = x['progen_input_ids']
        bs, set_size, seq_len = input_ids.shape
        
        input_ids = input_ids.view(bs * set_size, seq_len)
        shift_logits = self.model(input_ids, latent)[:, :-1, :]

        shift_labels = input_ids[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        return loss

    def sample(self, latent, num_samples=1, return_texts=False):
        device = latent.device
        batch_size = latent.size(0)
        start_ids = torch.tensor(
            [self.tokenizer.encode('1')] * batch_size,
            device=device
        )

        all_samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = self._generate(start_ids, latent)
            all_samples.append(out)

        out = torch.stack(all_samples, dim=1)

        if return_texts:
            texts = [
                [self.tokenizer.decode(out[b, n], skip_special_tokens=True)
                 for n in range(num_samples)]
                for b in range(batch_size)
            ]
            return out, texts

        return out

    def _generate(self, input_ids, latent):
        cur_ids = input_ids
        for _ in range(self.max_length - 2):
            with torch.no_grad():
                logits = self.model(cur_ids, latent)
            next_logits = logits[:, -1, :] / self.temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
        return cur_ids