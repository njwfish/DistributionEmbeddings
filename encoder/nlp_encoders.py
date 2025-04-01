import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from encoder.encoders import DistributionEncoder

class BertFeatureExtractor(nn.Module):
    """Module that extracts features from documents using BERT."""
    
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        output_dim=768,
        pooling_strategy="cls",
        freeze_bert=False,
    ):
        """
        Initialize the BERT feature extractor.
        
        Args:
            bert_model_name: Pretrained BERT model to use
            output_dim: Output dimension after any projection
            pooling_strategy: How to pool BERT outputs ('cls', 'mean', or 'max')
            freeze_bert: Whether to freeze BERT weights
        """
        super().__init__()
        
        # Initialize BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.pooling_strategy = pooling_strategy
        
        # Add projection if output_dim doesn't match BERT hidden size
        bert_hidden_size = self.bert.config.hidden_size
        if output_dim != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through BERT and pooling."""
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool based on strategy
        if self.pooling_strategy == "cls":
            # Use CLS token representation
            pooled = outputs.last_hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_strategy == "max":
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            pooled = torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Apply projection if needed
        pooled = self.projection(pooled)
        
        return pooled

class DocumentSetEncoder(nn.Module):
    """Encoder for sets of documents, combining BERT with distribution encoders."""
    
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        bert_output_dim=768,
        latent_dim=32,
        hidden_dim=128,
        pooling_strategy="cls",
        freeze_bert=False,
        distribution_encoder_type="tx",
        distribution_layers=2,
        distribution_heads=4,
    ):
        """
        Initialize the document set encoder.
        
        Args:
            bert_model_name: Pretrained BERT model to use
            bert_output_dim: Output dimension for BERT features
            latent_dim: Final latent dimension for distribution embedding
            hidden_dim: Hidden dimension for distribution encoder
            pooling_strategy: How to pool BERT outputs
            freeze_bert: Whether to freeze BERT weights
            distribution_encoder_type: Type of distribution encoder ('tx', 'gnn', or 'median_gnn')
            distribution_layers: Number of layers in distribution encoder
            distribution_heads: Number of heads for transformer distribution encoder
        """
        super().__init__()
        
        # BERT feature extractor
        self.bert_extractor = BertFeatureExtractor(
            bert_model_name=bert_model_name,
            output_dim=bert_output_dim,
            pooling_strategy=pooling_strategy,
            freeze_bert=freeze_bert
        )
        
        # Distribution encoder
        if distribution_encoder_type == "tx":
            from encoder.encoders import DistributionEncoderTx
            self.distribution_encoder = DistributionEncoderTx(
                in_dim=bert_output_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                set_size=None,  # Not used in forward pass
                layers=distribution_layers,
                heads=distribution_heads
            )
        elif distribution_encoder_type == "gnn":
            from encoder.encoders import DistributionEncoderGNN
            self.distribution_encoder = DistributionEncoderGNN(
                in_dim=bert_output_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                set_size=None,  # Not used in forward pass
                layers=distribution_layers,
                fc_layers=2
            )
        elif distribution_encoder_type == "median_gnn":
            from encoder.encoders import DistributionEncoderMedianGNN
            self.distribution_encoder = DistributionEncoderMedianGNN(
                in_dim=bert_output_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                set_size=None,  # Not used in forward pass
                layers=distribution_layers,
                fc_layers=2
            )
        else:
            raise ValueError(f"Unknown distribution encoder type: {distribution_encoder_type}")
    
    def forward(self, samples):
        """
        Forward pass through document set encoder.
        
        Args:
            samples: Dictionary containing 'bert_input_ids' and 'bert_attention_mask'
                    Shape: (batch_size, set_size, seq_len)
        
        Returns:
            Latent distribution embedding (batch_size, latent_dim)
        """
        batch_size, set_size = samples['bert_input_ids'].shape[:2]
        
        # Reshape for BERT processing
        input_ids = samples['bert_input_ids'].view(batch_size * set_size, -1)
        attention_mask = samples['bert_attention_mask'].view(batch_size * set_size, -1)
        
        # Extract BERT features
        doc_features = self.bert_extractor(input_ids, attention_mask)
        
        # Reshape for distribution encoder
        doc_features = doc_features.view(batch_size, set_size, -1)
        
        # Apply distribution encoder
        latent = self.distribution_encoder(doc_features)
        
        return latent 