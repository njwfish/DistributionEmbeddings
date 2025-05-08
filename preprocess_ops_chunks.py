import os
import numpy as np
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
from glob import glob
import shutil
import time

def preprocess_ops_by_perturbation(
    data_dir="/orcd/scratch/bcs/001/njwfish/data/ops", # "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/ops",
    metadata_file="ops_metadata.json",
    output_dir="by_perturbation",
    chunk_size=1000,
    checkpoint_interval=10  # Save progress every N perturbations
):
    print(f"Preprocessing OPS dataset into chunks by perturbation...")
    
    # Create output directory
    chunked_dir = os.path.join(data_dir, output_dir)
    os.makedirs(chunked_dir, exist_ok=True)
    
    # Check for existing index to resume processing
    index_path = os.path.join(chunked_dir, 'index.pkl')
    index = {}
    processed_perts = set()
    
    if os.path.exists(index_path):
        print(f"Found existing index file. Resuming from previous run...")
        try:
            with open(index_path, 'rb') as f:
                index = pickle.load(f)
            processed_perts = set(index.keys())
            print(f"Already processed {len(processed_perts)} perturbations")
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Starting fresh...")
    
    # Load metadata
    metadata_path = os.path.join(data_dir, metadata_file)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get all perturbations to process
    all_perts = set(metadata['pert_indices'].keys())
    perts_to_process = all_perts - processed_perts
    print(f"Found {len(all_perts)} total perturbations, {len(perts_to_process)} remaining to process")
    
    # Save checkpoint function
    def save_checkpoint():
        temp_path = index_path + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(index, f)
        # Atomic replace to prevent corruption
        os.replace(temp_path, index_path)
    
    # Track progress for checkpointing
    perts_since_checkpoint = 0
    
    # Process each perturbation
    for pert_idx, pert in enumerate(tqdm(sorted(perts_to_process), desc="Processing perturbations")):
        if pert in processed_perts:
            continue  # Skip already processed
            
        print(f"Processing perturbation: {pert} ({pert_idx+1}/{len(perts_to_process)})")
        pert_data = metadata['pert_indices'][pert]
        
        # Collect all images for this perturbation
        all_images = []
        
        for tile_entry in tqdm(pert_data, desc=f"Collecting {pert} images", leave=False):
            tile_idx = tile_entry['tile_idx']
            local_indices = np.array(tile_entry['local_indices'])
            
            if len(local_indices) == 0:
                continue
                
            tile_path = metadata['tile_paths'][tile_idx]
            try:
                # Load only the needed images
                images = np.load(f"{tile_path}/single_cell_images.npy")[local_indices]
                all_images.append(images)
            except Exception as e:
                print(f"Error loading images from {tile_path}: {e}")
                continue
        
        if not all_images:
            print(f"No images found for perturbation {pert}, skipping")
            # Mark as processed even if empty
            index[pert] = {'total_images': 0, 'chunks': [], 'chunk_size': chunk_size}
            processed_perts.add(pert)
            continue
            
        try:
            # Concatenate all images for this perturbation
            all_images = np.vstack(all_images)
            total_images = len(all_images)
            print(f"Total images for {pert}: {total_images}")
            
            # Save in chunks
            chunk_files = []
            for i in range(0, total_images, chunk_size):
                end = min(i + chunk_size, total_images)
                chunk = all_images[i:end]
                
                # Make sure directory exists
                pert_dir = os.path.join(chunked_dir, pert)
                os.makedirs(pert_dir, exist_ok=True)
                
                # Just save directly - simple and reliable
                chunk_file = os.path.join(pert_dir, f"chunk_{i//chunk_size}.npy")
                np.save(chunk_file, chunk)
                chunk_files.append(os.path.basename(chunk_file))
                
            # Store metadata for this perturbation
            index[pert] = {
                'total_images': total_images,
                'chunks': chunk_files,
                'chunk_size': chunk_size,
                'processed_at': time.time()
            }
            processed_perts.add(pert)
            
            # Save checkpoint periodically
            perts_since_checkpoint += 1
            if perts_since_checkpoint >= checkpoint_interval:
                print(f"Saving checkpoint after processing {perts_since_checkpoint} perturbations...")
                save_checkpoint()
                perts_since_checkpoint = 0
                
        except Exception as e:
            print(f"Error processing perturbation {pert}: {e}")
            # Save checkpoint on error to not lose progress
            save_checkpoint()
    
    # Save the final index
    save_checkpoint()
        
    print(f"Preprocessing complete. Data saved to {chunked_dir}")
    print(f"Total perturbations processed: {len(index)}")
    
    return chunked_dir, index

if __name__ == "__main__":
    chunked_dir, index = preprocess_ops_by_perturbation()
    
    # Print some stats
    print("\nPerturbation statistics:")
    for pert, info in sorted(index.items(), key=lambda x: x[1]['total_images'], reverse=True):
        print(f"{pert}: {info['total_images']} images in {len(info['chunks'])} chunks") 