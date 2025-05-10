import torch
import torch.nn as nn
from encoder.dna_conv_encoder import DNAConvEncoder

class DNAConvEncoderClassifier(DNAConvEncoder):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        latent_dim = kwargs['latent_dim']
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_classes),
            nn.Softmax(dim=1)
        )
        
