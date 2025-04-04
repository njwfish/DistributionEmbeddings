import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import textwrap

def visualize_data(save_path, real, generated):
    original_flat = real.numpy()
    generated_flat = generated.numpy()

    # scatter plot the first and second moments of the original and generated data
    original_mean = np.mean(original_flat, axis=0)
    original_std = np.std(original_flat, axis=0)
    generated_mean = np.mean(generated_flat, axis=0)
    generated_std = np.std(generated_flat, axis=0)

    # plot the first and second moments of the original and generated data
    r2_mean = np.corrcoef(original_mean, generated_mean)[0, 1]**2
    r2_std = np.corrcoef(original_std, generated_std)[0, 1]**2
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].scatter(original_mean, generated_mean)
    axes[0].set_title("Original Mean vs Generated Mean") 
    # put text in the top left of the plot
    axes[0].text(0.05, 0.95, f"R2: {r2_mean:.2f}", ha='left', va='top', transform=axes[0].transAxes)
    axes[1].scatter(original_std, generated_std)
    axes[1].set_title("Original Std vs Generated Std")
    axes[1].text(0.05, 0.95, f"R2: {r2_std:.2f}", ha='left', va='top', transform=axes[1].transAxes)
    # add axis labels
    axes[0].set_xlabel("Original Mean")
    axes[0].set_ylabel("Generated Mean")
    axes[1].set_xlabel("Original Std")
    axes[1].set_ylabel("Generated Std")
    plt.savefig(save_path)
    plt.close()

    