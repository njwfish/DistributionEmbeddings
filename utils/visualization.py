import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import textwrap

def visualize_data(save_path, real, generated, max_features_to_plot=10):
    if len(real.shape) == 2: # [set_size, features]
        original_flat = real.numpy()
        generated_flat = generated.numpy()
        _, features = real.shape

        if features > max_features_to_plot:
            random_features = np.random.choice(features, size=max_features_to_plot, replace=False)
            original_flat = original_flat[:, random_features]
            generated_flat = generated_flat[:, random_features]
        
        # Create a pairplot of the original and generated data
        # save to outputs config directory
        df = pd.DataFrame(np.concatenate([original_flat, generated_flat], axis=0))
        df['hue'] = ['original'] * len(original_flat) + ['generated'] * len(generated_flat)
        sns.pairplot(df, hue='hue')
        plt.savefig(save_path)
        plt.close()

    elif len(real.shape) == 3: # [set_size, channels, height, width]

        # Create grids of real and generated images
        real_grid = make_grid(real*-1 + 1, nrow=10)
        gen_grid = make_grid(generated*-1 + 1, nrow=10)
        
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot real images on the left
        axes[0].imshow(real_grid.permute(1, 2, 0).cpu().numpy())
        axes[0].axis('off')
        axes[0].set_title("Original Samples")
        
        # Plot generated images on the right
        axes[1].imshow(gen_grid.permute(1, 2, 0).cpu().numpy())
        axes[1].axis('off')
        axes[1].set_title("Generated Samples")
        
        # Add a main title for the entire figure
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path)
        plt.close()

def visualize_text_data(output_dir, original_texts, generated_texts):
    """
    Visualize original and generated text data from the PubMed dataset.
    
    Args:
        output_dir: Directory to save the visualizations
        original_texts: Original texts from the dataset
        generated_texts: Generated texts from the model
        max_examples: Maximum number of examples to visualize
        max_texts_per_example: Maximum number of texts per example
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(len(original_texts)):
        df_original = pd.DataFrame(original_texts[i])
        df_original['type'] = 'original'
        df_generated = pd.DataFrame(generated_texts[i])
        df_generated['type'] = 'generated'
        df_combined = pd.concat([df_original, df_generated], axis=0)
        df_combined.to_csv(os.path.join(output_dir, f"text_samples_{i}.csv"), index=False)
    
    