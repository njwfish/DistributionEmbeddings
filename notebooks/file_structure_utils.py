import os
import pandas as pd
import numpy as np
from skimage import io
from glob import glob

def analyze_plate_structure(base_dir='../data/ops', screen='screenA'):
    """
    Analyze the structure of plates in the dataset.
    
    Args:
        base_dir: Base directory where data is stored
        screen: Screen name (e.g., 'screenA')
        
    Returns:
        DataFrame with plate structure information
    """
    plates = []
    plate_paths = glob(os.path.join(base_dir, screen, '*/'))
    
    for plate_path in plate_paths:
        plate_name = os.path.basename(os.path.dirname(plate_path))
        
        # Find channel directories
        channel_dir_path = os.path.join(plate_path, 'phenotype', 'images', 'input')
        if not os.path.exists(channel_dir_path):
            continue
            
        channel_dirs = os.listdir(channel_dir_path)
        
        for channel_dir in channel_dirs:
            # Check channel patterns in this directory
            full_channel_dir = os.path.join(channel_dir_path, channel_dir)
            if not os.path.isdir(full_channel_dir):
                continue
                
            # Sample a few files to identify channel patterns
            sample_files = glob(os.path.join(full_channel_dir, '*.tif'))[:5]
            channel_patterns = set()
            
            for file in sample_files:
                filename = os.path.basename(file)
                # Extract the channel pattern (like A594, AF750, DAPI-GFP)
                parts = filename.split('_')
                if len(parts) >= 4:
                    channel_patterns.add(parts[3])
            
            plates.append({
                'plate': plate_name,
                'channel_dir': channel_dir,
                'channel_patterns': ', '.join(channel_patterns)
            })
    
    return pd.DataFrame(plates)

def get_image_path(plate, well, site, channel, base_dir='../data/ops', screen='screenA'):
    """
    Construct the path to an image based on metadata.
    
    Args:
        plate: Plate name (e.g., '20200202_6W-LaC024A')
        well: Well ID (e.g., 'A1')
        site: Site number
        channel: Channel name ('DAPI-GFP', 'A594', or 'AF750')
        base_dir: Base directory where data is stored
        screen: Screen name
        
    Returns:
        Full path to the image file or None if not found
    """
    # Check which directory structure this plate has
    plate_path = os.path.join(base_dir, screen, plate)
    input_dir = os.path.join(plate_path, 'phenotype', 'images', 'input')
    
    if not os.path.exists(input_dir):
        return None
    
    # Find channel directories
    channel_dirs = os.listdir(input_dir)
    
    for channel_dir in channel_dirs:
        channel_dir_path = os.path.join(input_dir, channel_dir)
        
        # Construct possible file patterns based on the channel directory name
        if channel_dir == 'DAPI-GFP-A594-AF750':
            file_pattern = f"20X_DAPI-GFP-A594-AF750_{well}_{channel}_Site-{site}.tif"
        elif channel_dir == 'A594-AF750' and channel in ['A594', 'AF750']:
            file_pattern = f"20X_A594-AF750_{well}_{channel}_Site-{site}.tif"
        else:
            # Try a generic pattern if none of the above match
            file_pattern = f"*_{well}_{channel}_Site-{site}.tif"
        
        # Check if the file exists
        file_path = os.path.join(channel_dir_path, file_pattern)
        matching_files = glob(file_path)
        
        if matching_files:
            return matching_files[0]
    
    return None

def get_available_channels(plate, well, site, base_dir='../data/ops', screen='screenA'):
    """
    Find which channels are available for a given plate, well, and site.
    
    Args:
        plate: Plate name
        well: Well ID
        site: Site number
        base_dir: Base directory where data is stored
        screen: Screen name
        
    Returns:
        List of available channels
    """
    channels = ['DAPI-GFP', 'A594', 'AF750']
    available_channels = []
    
    for channel in channels:
        path = get_image_path(plate, well, site, channel, base_dir, screen)
        if path and os.path.exists(path):
            available_channels.append(channel)
    
    return available_channels

def extract_cell_from_site(plate, well, site, cell_bounds, channels=None, 
                          pad=5, base_dir='../data/ops', screen='screenA'):
    """
    Extract a cell image from specified site using bounding box coordinates.
    
    Args:
        plate: Plate name
        well: Well ID
        site: Site number
        cell_bounds: List/tuple of [min_row, min_col, max_row, max_col]
        channels: List of channels to extract (if None, will get all available)
        pad: Number of pixels to pad around the bounding box
        base_dir: Base directory where data is stored
        screen: Screen name
        
    Returns:
        Dictionary of numpy arrays for each channel
    """
    # Get bounding box coordinates
    min_row, min_col, max_row, max_col = [int(x) for x in cell_bounds]
    
    # Add padding
    min_row = max(0, min_row - pad)
    min_col = max(0, min_col - pad)
    max_row = max_row + pad
    max_col = max_col + pad
    
    # Find available channels if not specified
    if channels is None:
        channels = get_available_channels(plate, well, site, base_dir, screen)
    
    # Extract images for each channel
    cell_images = {}
    
    for channel in channels:
        try:
            # Get full image path
            image_path = get_image_path(plate, well, site, channel, base_dir, screen)
            
            # Check if file exists
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: File not found for channel {channel}")
                continue
                
            # Load full image
            full_image = io.imread(image_path)
            
            # Debug: print shape information
            print(f"Channel {channel} image shape: {full_image.shape}")
            
            # Handle different channel formats based on observed shapes
            if channel == 'DAPI-GFP':
                # Based on the observed shape (4, 2, 2960, 2960)
                # This appears to be a 4D array with structure:
                # (time points, channels, height, width)
                if len(full_image.shape) == 4 and full_image.shape[1] == 2:
                    # Extract DAPI (take first time point, first channel)
                    dapi_image = full_image[:, 0]
                    img = dapi_image[:, min_row:max_row, min_col:max_col]
                    # transpose the image to be (height, width, channels)
                    cell_images['DAPI'] = np.transpose(img, (1, 2, 0))
                    
                    # Extract GFP (take first time point, second channel)
                    gfp_image = full_image[:, 1] 
                    img = gfp_image[:, min_row:max_row, min_col:max_col]
                    # transpose the image to be (height, width, channels)
                    cell_images['GFP'] = np.transpose(img, (1, 2, 0))
                else:
                    raise ValueError(f"Unexpected image shape for channel {channel}: {full_image.shape}")
            else:
                # For A594 and AF750 with shape (2960, 2960, 4)
                # This appears to be RGB + alpha channel format
                if len(full_image.shape) == 3 and full_image.shape[2] == 4:
                    # Extract RGB data, ignoring alpha channel
                    # For consistency in later processing, just take the RED channel (index 0)
                    # as these are grayscale images stored in RGBA format
                    cell_images[channel] = full_image[min_row:max_row, min_col:max_col]
                else:
                    raise ValueError(f"Unexpected image shape for channel {channel}: {full_image.shape}")
            
        except Exception as e:
            print(f"Error processing {channel} channel: {e}")
            import traceback
            traceback.print_exc()
    
    return cell_images

def create_rgb_cell_image(cell_images, channel_mapping=None):
    """
    Create an RGB image from individual channel images.
    
    Args:
        cell_images: Dictionary of images for each channel
        channel_mapping: Dictionary mapping channel names to RGB colors
                         If None, defaults to DAPI=blue, GFP=green, A594=red
        
    Returns:
        RGB image as numpy array
    """
    if channel_mapping is None:
        channel_mapping = {
            'DAPI': 'blue',
            'GFP': 'green',
            'A594': 'red',
            'AF750': 'yellow'
        }
    
    # Find the shape of the cell image
    shape = None
    for img in cell_images.values():
        if shape is None:
            # Ensure we're only using the first two dimensions for shape calculation
            if len(img.shape) > 2:
                shape = img.shape[:2]
            else:
                shape = img.shape
        else:
            current_shape = img.shape[:2] if len(img.shape) > 2 else img.shape
            # Make sure all images have the same shape
            if current_shape != shape:
                from skimage import transform
                # Only resize if the image is 2D
                if len(img.shape) == 2:
                    img = transform.resize(img, shape, preserve_range=True).astype(img.dtype)
    
    if shape is None:
        return None
    
    # Create empty RGB image
    rgb_image = np.zeros((*shape, 3), dtype=np.float32)
    
    # Normalize intensity for each channel
    for channel, color in channel_mapping.items():
        if channel in cell_images:
            img = cell_images[channel].astype(np.float32)
            
            # Handle RGB inputs by taking the first channel only
            if len(img.shape) > 2:
                if img.shape[2] == 3:  # RGB
                    # Just use the first channel for simplicity
                    img = img[:, :, 0]
                elif img.shape[2] == 4:  # RGBA
                    # Use the first channel and ignore alpha
                    img = img[:, :, 0]
            
            # Normalize to [0, 1]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            
            # Add to the appropriate color channel
            if color == 'red':
                rgb_image[:, :, 0] += img
            elif color == 'green':
                rgb_image[:, :, 1] += img
            elif color == 'blue':
                rgb_image[:, :, 2] += img
    
    # Clip values to [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image 