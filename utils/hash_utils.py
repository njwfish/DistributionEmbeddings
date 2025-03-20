import hashlib
import json
import os
from typing import Dict, Any, Union
import yaml
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

def hash_config(config: Union[Dict[str, Any], DictConfig], include_keys: list = None, exclude_keys: list = None) -> str:
    """
    Generate a deterministic hash from a configuration dictionary or DictConfig object.
    
    Args:
        config: Configuration dictionary or DictConfig object
        include_keys: If provided, only these keys will be considered for hashing
        exclude_keys: If provided, these keys will be excluded from hashing
        
    Returns:
        String hash representation of the config
    """
    # Convert OmegaConf to dict if needed
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config.copy()
    
    # Filter keys if needed
    filtered_config = {}
    
    if include_keys is not None:
        for key in include_keys:
            if key in config_dict:
                filtered_config[key] = config_dict[key]
    else:
        filtered_config = config_dict.copy()
        
    if exclude_keys is not None:
        for key in exclude_keys:
            if key in filtered_config:
                del filtered_config[key]
    
    # Remove non-deterministic or irrelevant keys
    default_exclude = ['hydra', 'seed', 'device', 'wandb', 'output_dir']
    for key in default_exclude:
        if key in filtered_config:
            del filtered_config[key]
    
    # Sort keys for deterministic ordering
    sorted_dict = json.dumps(filtered_config, sort_keys=True)
    
    # Generate hash
    hash_obj = hashlib.md5(sorted_dict.encode())
    config_hash = hash_obj.hexdigest()[:10]  # Use first 10 chars for readability
    
    return config_hash

def get_output_dir(config: Union[Dict[str, Any], DictConfig], 
                  base_dir: str = './outputs',
                  experiment_name: str = None,
                  create_dir: bool = True) -> str:
    """
    Get a deterministic output directory based on config hash.
    
    Args:
        config: Configuration dictionary or DictConfig
        base_dir: Base directory for outputs
        experiment_name: Optional experiment name to prepend to hash
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Path to output directory
    """
    config_hash = hash_config(config)
    
    if experiment_name is None:
        if isinstance(config, DictConfig) and 'experiment_name' in config:
            experiment_name = config.experiment_name
        else:
            experiment_name = 'experiment'
    
    # Create directory name from experiment name and hash
    dir_name = f"{experiment_name}_{config_hash}"
    output_dir = os.path.join(base_dir, dir_name)
    
    # Create directory if it doesn't exist
    if create_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Save config for reference
        if isinstance(config, DictConfig):
            config_path = os.path.join(output_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(config))
        else:
            config_path = os.path.join(output_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    return output_dir

def find_matching_output_dir(config: Union[Dict[str, Any], DictConfig], 
                           base_dir: str = './outputs') -> str:
    """
    Find an existing output directory with a matching config hash.
    
    Args:
        config: Configuration to match
        base_dir: Base directory for outputs
        
    Returns:
        Path to matching directory or None if not found
    """
    config_hash = hash_config(config)
    
    if not os.path.exists(base_dir):
        return None
        
    for dir_name in os.listdir(base_dir):
        if dir_name.endswith(config_hash):
            return os.path.join(base_dir, dir_name)
    
    return None 