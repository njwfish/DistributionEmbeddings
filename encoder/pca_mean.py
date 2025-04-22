import torch
import torch.nn as nn
from torchvision import datasets, transforms

def mnist_pca_matrix(n_pcs=32):
    # load mnist train data and flatten each image
    mnist = datasets.MNIST(root='.', train=True, download=True,
                           transform=transforms.ToTensor())
    data = torch.stack([img.view(-1) for img, _ in mnist])  # shape (60000, 784)

    # center the data
    mean = data.mean(dim=0)  # shape (784,)
    centered = data - mean

    # low-rank PCA
    U, S, V = torch.pca_lowrank(centered, q=n_pcs)  # V: (784, n_pcs)

    return V, mean  # projection matrix and mean for centering

class MNISTPCAEncoder(nn.Module):
    def __init__(self, n_pcs=32, device='cuda'):
        super().__init__()

        # precompute pca from mnist !
        proj_matrix, mean_vector = mnist_pca_matrix(n_pcs)

        # save as fixed tensors (not trainable !)
        self.proj = proj_matrix.to(device)  # (784, n_pcs)
        self.mean = mean_vector.to(device)  # (784,)

    def forward(self, x):
        # x: (batch, set_size, 1, 28, 28)

        x = x.flatten(start_dim=2)

        # center each element in the set
        x_centered = x - self.mean

        # apply pca projection to each vector in the set
        # proj: (784, n_pcs)
        x_proj = torch.matmul(x_centered, self.proj)  # (batch, set_size, n_pcs)

        # take mean across the set dimension
        x_mean = x_proj.mean(dim=1)  # (batch, n_pcs)

        return x_mean