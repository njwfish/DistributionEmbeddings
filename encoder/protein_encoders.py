from transformers import EsmModel
from encoder.encoders import DistributionEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESMFeatureExtractor(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", output_dim=320, pooling="mean", freeze=False):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        if freeze:
            for p in self.esm.parameters(): p.requires_grad = False
        self.pooling = pooling
        h = self.esm.config.hidden_size
        self.proj = nn.Linear(h, output_dim) if output_dim != h else nn.Identity()

    def forward(self, input_ids, attention_mask=None):
        x = self.esm(input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            x[mask == 0] = -1e9
            pooled = x.max(1).values
        else:
            raise ValueError("bad pooling :(")
        return self.proj(pooled)

class ProteinSetEncoder(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", 
                 esm_dim=320, latent_dim=32, hidden_dim=128,
                 pooling="mean", freeze=False, dist_type="tx", layers=2, heads=4):
        super().__init__()
        self.esm_extractor = ESMFeatureExtractor(esm_model_name, esm_dim, pooling, freeze)
        if dist_type == "tx":
            from encoder.encoders import DistributionEncoderTx as DE
            self.dist = DE(esm_dim, latent_dim, hidden_dim, None, layers, heads)
        elif dist_type == "gnn":
            from encoder.encoders import DistributionEncoderGNN as DE
            self.dist = DE(esm_dim, latent_dim, hidden_dim, None, layers, fc_layers=2)
        elif dist_type == "median_gnn":
            from encoder.encoders import DistributionEncoderMedianGNN as DE
            self.dist = DE(esm_dim, latent_dim, hidden_dim, None, layers, fc_layers=2)
        else:
            raise ValueError("bad dist type :(")

    def forward(self, samples):
        b, s = samples['esm_input_ids'].shape[:2]
        ids = samples['esm_input_ids'].view(b * s, -1)
        mask = samples['esm_attention_mask'].view(b * s, -1)
        feats = self.esm_extractor(ids, mask).view(b, s, -1)
        return self.dist(feats)