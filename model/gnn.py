import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLP

class GNN(nn.Module):
    def __init__(
            self, gnn_dim, in_dims, hidden_dim, 
            out_dim=1, embedding_dim=64, shared_embedding_dim=128, layers=2, layers_shared=2
    ):
        super(GNN, self).__init__()
        self.gnn_dim = gnn_dim
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.shared_mlp = MLP(
            [gnn_dim] + in_dims, hidden_dim * 10, shared_embedding_dim, layers=layers_shared
        )

        self.mlp = MLP(
            [1, embedding_dim, shared_embedding_dim] + in_dims, hidden_dim, out_dim, layers=layers
        )
        self.positional_embedding = torch.nn.Embedding(num_embeddings=gnn_dim, embedding_dim=embedding_dim)

    def forward(self, *x, subsample_indices=None):
        # indices = torch.randperm(self.gnn_dim)[:num_nodes].to(x[0].device)
        if subsample_indices is None:
            subsample_indices = torch.arange(self.gnn_dim).to(x[0].device)

        nodes = x[0][:, subsample_indices]
        nodes = nodes.unsqueeze(2)

        # get a random subset of indices
        z = self.positional_embedding(subsample_indices)  # shape: [gnn_dim, embedding_dim]
        z = z.unsqueeze(0).repeat(x[0].shape[0], 1, 1)

        shared_embedding = self.shared_mlp(*x)
        shared_embedding = shared_embedding.unsqueeze(1).repeat(1, nodes.shape[1], 1)

        expanded_x = [xi.unsqueeze(1).repeat(1, nodes.shape[1], 1) for xi in x[1:]]

        out = torch.vmap(self.mlp, in_dims=(1, 1, 1, 1, 1))(nodes, z, shared_embedding, *expanded_x)
        out = out.squeeze()
        out = out.t()
        return out