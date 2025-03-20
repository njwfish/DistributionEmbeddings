import os
import json
import yaml
import torch
import logging
from utils.hash_utils import hash_config, find_matching_output_dir
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Union, List, Optional, Tuple

logger = logging.getLogger(__name__)

def find_all_experiments(base_dir: str = './outputs') -> List[str]:
    """Find all experiment directories in the base directory."""
    if not os.path.exists(base_dir):
        return []
    
    return [d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) and '_' in d]

def get_experiment_info(experiment_dir: str) -> Dict[str, Any]:
    """
    Get information about an experiment, including its config and results.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary with experiment information
    """
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory {experiment_dir} not found")
    
    info = {
        'dir': experiment_dir,
        'name': os.path.basename(experiment_dir),
        'config': None,
        'best_model': None,
        'checkpoints': [],
        'config_hash': None
    }
    
    # Extract config hash from directory name
    parts = info['name'].split('_')
    if len(parts) > 1:
        info['config_hash'] = parts[-1]
    
    # Find config file
    config_yaml = os.path.join(experiment_dir, 'config.yaml')
    config_json = os.path.join(experiment_dir, 'config.json')
    
    if os.path.exists(config_yaml):
        with open(config_yaml, 'r') as f:
            info['config'] = OmegaConf.load(config_yaml)
    elif os.path.exists(config_json):
        with open(config_json, 'r') as f:
            info['config'] = json.load(f)
    
    # Find best model
    best_model_path = os.path.join(experiment_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        info['best_model'] = best_model_path
        # Try to get information from the checkpoint
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            info['best_epoch'] = checkpoint.get('epoch')
            info['best_loss'] = checkpoint.get('loss')
        except:
            logger.warning(f"Could not load checkpoint from {best_model_path}")
    
    # Find checkpoints
    for filename in os.listdir(experiment_dir):
        if filename.startswith('checkpoint_epoch_'):
            info['checkpoints'].append(os.path.join(experiment_dir, filename))
    
    info['checkpoints'].sort()  # Sort by filename (which includes epoch)
    
    return info

def find_experiments_by_config(config: Union[Dict[str, Any], DictConfig], 
                               base_dir: str = './outputs',
                               exact_match: bool = True) -> List[str]:
    """
    Find experiments with matching configs.
    
    Args:
        config: Configuration to match
        base_dir: Base directory for outputs
        exact_match: Whether to require an exact config match
        
    Returns:
        List of matching experiment directories
    """
    if exact_match:
        # For exact matches, we can use the hash
        matching_dir = find_matching_output_dir(config, base_dir)
        return [matching_dir] if matching_dir else []
    
    # For partial matches, we need to load and compare configs
    matches = []
    
    for exp_dir_name in find_all_experiments(base_dir):
        exp_dir = os.path.join(base_dir, exp_dir_name)
        exp_info = get_experiment_info(exp_dir)
        
        if exp_info['config'] is not None:
            # Check if the experiment config contains all the provided config values
            config_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
            exp_config = OmegaConf.to_container(exp_info['config'], resolve=True) if isinstance(exp_info['config'], DictConfig) else exp_info['config']
            
            matches_all = True
            for key, value in config_dict.items():
                if key not in exp_config or exp_config[key] != value:
                    matches_all = False
                    break
            
            if matches_all:
                matches.append(exp_dir)
    
    return matches

def load_best_model(experiment_dir: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load the best model from an experiment.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Tuple of (model checkpoint, path to the checkpoint file)
    """
    best_model_path = os.path.join(experiment_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        logging.warning(f"No best model found in {experiment_dir}")
        return None, None
    
    try:
        checkpoint = torch.load(best_model_path, map_location='cpu')
        return checkpoint, best_model_path
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return None, best_model_path

def compare_experiments(exp_dir1: str, exp_dir2: str) -> Dict[str, Any]:
    """
    Compare two experiments and return their differences.
    
    Args:
        exp_dir1: Path to the first experiment directory
        exp_dir2: Path to the second experiment directory
        
    Returns:
        Dictionary with differences between the experiments
    """
    exp1 = get_experiment_info(exp_dir1)
    exp2 = get_experiment_info(exp_dir2)
    
    if exp1['config'] is None or exp2['config'] is None:
        return {'error': 'One or both experiments do not have a config file'}
    
    # Convert configs to dictionaries for comparison
    cfg1 = OmegaConf.to_container(exp1['config'], resolve=True) if isinstance(exp1['config'], DictConfig) else exp1['config']
    cfg2 = OmegaConf.to_container(exp2['config'], resolve=True) if isinstance(exp2['config'], DictConfig) else exp2['config']
    
    # Find config differences
    config_diffs = {}
    all_keys = set(cfg1.keys()) | set(cfg2.keys())
    
    for key in all_keys:
        if key not in cfg1:
            config_diffs[key] = (None, cfg2[key])
        elif key not in cfg2:
            config_diffs[key] = (cfg1[key], None)
        elif cfg1[key] != cfg2[key]:
            config_diffs[key] = (cfg1[key], cfg2[key])
    
    # Compare performance metrics
    perf_diffs = {}
    metrics = ['best_loss', 'best_epoch']
    
    for metric in metrics:
        val1 = exp1.get(metric)
        val2 = exp2.get(metric)
        
        if val1 is not None and val2 is not None and val1 != val2:
            perf_diffs[metric] = (val1, val2)
    
    return {
        'exp1': os.path.basename(exp_dir1),
        'exp2': os.path.basename(exp_dir2),
        'config_differences': config_diffs,
        'performance_differences': perf_diffs
    } 