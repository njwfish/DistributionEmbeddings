import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import textwrap

def visualize_data(save_path, real, generated):
    if len(real.shape) == 3: # [batch, set_size, features]
        original_flat = real.numpy()
        generated_flat = generated.numpy()
        
        # Create a pairplot of the original and generated data
        # save to outputs config directory
        df = pd.DataFrame(np.concatenate([original_flat, generated_flat], axis=0))
        df['hue'] = ['original'] * len(original_flat) + ['generated'] * len(generated_flat)
        sns.pairplot(df, hue='hue')
        plt.savefig(save_path)
        plt.close()

    elif len(real.shape) == 4: # [batch, set_size, channels, height, width]

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

def visualize_text_data(output_dir, original_texts, generated_texts, max_examples=5, max_texts_per_example=3):
    """
    Visualize original and generated text data from the PubMed dataset.
    
    Args:
        output_dir: Directory to save the visualizations
        original_texts: Original texts from the dataset
        generated_texts: Generated texts from the model
        max_examples: Maximum number of examples to visualize
        max_texts_per_example: Maximum number of texts per example
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate inputs and handle possible errors
        if not original_texts or not generated_texts:
            print("Warning: Empty text data provided for visualization")
            # Create a placeholder visualization
            with open(os.path.join(output_dir, "text_samples.txt"), 'w') as f:
                f.write("No valid text samples available for visualization.\n")
            return
        
        # Ensure original_texts and generated_texts have consistent types
        if not isinstance(original_texts, list):
            print(f"Warning: original_texts is not a list, got {type(original_texts)}")
            original_texts = [[str(original_texts)]]
        elif original_texts and not isinstance(original_texts[0], list):
            original_texts = [original_texts]
            
        if not isinstance(generated_texts, list):
            print(f"Warning: generated_texts is not a list, got {type(generated_texts)}")
            generated_texts = [[str(generated_texts)]]
        elif generated_texts and not isinstance(generated_texts[0], list):
            generated_texts = [generated_texts]
        
        # Determine how many examples to visualize
        num_examples = min(len(original_texts), len(generated_texts), max_examples)
        
        for i in range(num_examples):
            try:
                # Get the original and generated texts for this example
                orig_set = original_texts[i]
                gen_set = generated_texts[i]
                
                # Ensure texts are strings
                orig_set = [str(text) for text in orig_set]
                gen_set = [str(text) for text in gen_set]
                
                # Limit the number of texts per example
                orig_set = orig_set[:max_texts_per_example]
                gen_set = gen_set[:max_texts_per_example]
                
                # Create a figure
                fig, ax = plt.subplots(figsize=(20, 10))
                ax.axis('off')
                
                # Add the original and generated texts
                text_content = "ORIGINAL TEXTS:\n\n"
                for j, text in enumerate(orig_set):
                    wrapped_text = textwrap.fill(text, width=100)
                    text_content += f"{j+1}. {wrapped_text}\n\n"
                
                text_content += "\nGENERATED TEXTS:\n\n"
                for j, text in enumerate(gen_set):
                    wrapped_text = textwrap.fill(text, width=100)
                    text_content += f"{j+1}. {wrapped_text}\n\n"
                
                ax.text(0.05, 0.95, text_content, va='top', fontsize=12, 
                        fontfamily='monospace', linespacing=1.5)
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"text_comparison_{i}.png"))
                plt.close()
            except Exception as e:
                print(f"Error visualizing example {i}: {e}")
                continue
        
        # Also save the texts as a plain text file for easier reading
        with open(os.path.join(output_dir, "text_samples.txt"), 'w') as f:
            for i in range(num_examples):
                try:
                    f.write(f"Sample {i+1}\n")
                    f.write("="*80 + "\n\n")
                    
                    f.write("ORIGINAL TEXTS:\n")
                    orig_set = original_texts[i][:max_texts_per_example]
                    for j, text in enumerate(orig_set):
                        f.write(f"{j+1}. {text}\n\n")
                    
                    f.write("\nGENERATED TEXTS:\n")
                    gen_set = generated_texts[i][:max_texts_per_example]
                    for j, text in enumerate(gen_set):
                        f.write(f"{j+1}. {text}\n\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
                except Exception as e:
                    f.write(f"Error writing sample {i+1}: {e}\n\n")
                    continue
    except Exception as e:
        print(f"Error in text visualization: {e}")
        # Create a basic error file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "visualization_error.txt"), 'w') as f:
            f.write(f"Error during text visualization: {e}\n")
            f.write("Original texts type: " + str(type(original_texts)) + "\n")
            f.write("Generated texts type: " + str(type(generated_texts)) + "\n")