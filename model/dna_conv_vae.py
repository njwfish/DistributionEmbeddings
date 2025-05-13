import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class DNAConvVAE(nn.Module):
    def __init__(
        self, 
        seq_len,
        vocab_size=5, 
        hidden_channels_list=[64, 128, 256], # Channels for each conv layer in encoder
        kernel_size=3,
        condition_latent_dim=32, 
        vae_latent_dim=128,
        embed_dim_for_condition=64
    ):
        super(DNAConvVAE, self).__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.condition_latent_dim = condition_latent_dim
        self.vae_latent_dim = vae_latent_dim
        self.embed_dim_for_condition = embed_dim_for_condition

        # Condition embedding
        self.condition_embed = EmbedFC(condition_latent_dim, embed_dim_for_condition)

        # Encoder
        encoder_layers = []
        current_channels = vocab_size
        current_seq_len = seq_len
        for H_out in hidden_channels_list:
            encoder_layers.append(nn.Conv1d(current_channels, H_out, kernel_size=kernel_size, padding=kernel_size // 2))
            encoder_layers.append(nn.BatchNorm1d(H_out))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            current_channels = H_out
            current_seq_len = current_seq_len // 2 # MaxPool1d with kernel 2, stride 2
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        self.encoder_output_dim = current_channels * current_seq_len
        
        self.fc_mu = nn.Linear(self.encoder_output_dim + embed_dim_for_condition, vae_latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim + embed_dim_for_condition, vae_latent_dim)

        # Decoder
        self.decoder_input_dim = current_channels * current_seq_len # This is where encoder_conv output shape matches
        self.fc_decode = nn.Linear(vae_latent_dim + embed_dim_for_condition, self.decoder_input_dim)

        decoder_layers = []
        # hidden_channels_list is [C1, C2, C3], current_channels is C3
        # We want to go C3 -> C2 -> C1 -> vocab_size
        reversed_hidden_channels = hidden_channels_list[::-1] # [C3, C2, C1]
        
        # Input to first ConvTranspose1d is current_channels (e.g. C3)
        # Output should be C2, then C1, then vocab_size
        
        for i in range(len(reversed_hidden_channels)):
            H_in = reversed_hidden_channels[i]
            if i + 1 < len(reversed_hidden_channels):
                H_out = reversed_hidden_channels[i+1]
            else: # Last layer, output should be vocab_size
                H_out = vocab_size
            
            decoder_layers.append(nn.ConvTranspose1d(H_in, H_out, kernel_size=4, stride=2, padding=1))
            # Don't add BatchNorm/GELU to the very last layer before softmax
            if i < len(reversed_hidden_channels) -1 :
                 decoder_layers.append(nn.BatchNorm1d(H_out))
                 decoder_layers.append(nn.GELU())
            elif H_out != vocab_size : # if the last element of reversed_hidden_channels is not vocab_size
                 decoder_layers.append(nn.BatchNorm1d(H_out)) # This case should ideally not be hit with current logic
                 decoder_layers.append(nn.GELU())


        self.decoder_conv = nn.Sequential(*decoder_layers)
        self.final_reduced_seq_len = current_seq_len


    def encode(self, x, c):
        # x shape from CVAE.forward: (batch_size_effective, seq_len, vocab_size)
        # Conv1d expects: (batch_size_effective, vocab_size, seq_len)
        x = x.permute(0, 2, 1)
        
        c_embedded = self.condition_embed(c) # (batch_size, embed_dim_for_condition)
        
        h_conv = self.encoder_conv(x) # (batch_size, last_hidden_channel, reduced_seq_len)
        h_flat = h_conv.view(h_conv.size(0), -1) # (batch_size, encoder_output_dim)
        
        h_concat = torch.cat([h_flat, c_embedded], dim=1)
        
        mu = self.fc_mu(h_concat)
        logvar = self.fc_logvar(h_concat)
        
        return mu, logvar

    def decode(self, z, c):
        # z shape: (batch_size, vae_latent_dim)
        # c shape: (batch_size, condition_latent_dim)

        c_embedded = self.condition_embed(c) # (batch_size, embed_dim_for_condition)
        
        z_concat = torch.cat([z, c_embedded], dim=1)
        
        h_fc_decode = self.fc_decode(z_concat) # (batch_size, decoder_input_dim)
        
        # Reshape to (batch_size, last_hidden_channel, final_reduced_seq_len)
        # last_hidden_channel is hidden_channels_list[-1]
        # final_reduced_seq_len is self.seq_len // (2**len(hidden_channels_list))
        
        # Calculate the channel size for reshaping based on encoder's last channel
        reshape_channels = self.encoder_conv[-4].out_channels # Accessing out_channels of last Conv1d in encoder
        
        h_reshaped = h_fc_decode.view(h_fc_decode.size(0), reshape_channels, self.final_reduced_seq_len)
        
        output_logits = self.decoder_conv(h_reshaped) # (batch_size, vocab_size, current_decoded_seq_len)

        # Ensure output sequence length matches self.seq_len (from config)
        if output_logits.shape[2] != self.seq_len:
            output_logits = F.interpolate(output_logits, size=self.seq_len, mode='linear', align_corners=False)
        
        # Permute to (batch_size, seq_len, vocab_size) for softmax as requested
        output_permuted = output_logits.permute(0, 2, 1)
        
        return F.softmax(output_permuted, dim=-1) 