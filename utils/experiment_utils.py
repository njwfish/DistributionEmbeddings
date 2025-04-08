import os
import glob
import json
import yaml
import torch
import logging
from utils.hash_utils import hash_config, find_matching_output_dir
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Dict, Any, Union, List, Optional, Tuple

torch.serialization.add_safe_globals([ListConfig])

logger = logging.getLogger(__name__)

def find_all_experiments(base_dir: str) -> List[str]:
    """
    Find all experiment directories in the base_dir.
    Returns a list of directory names (not full paths).
    """
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory {base_dir} does not exist")
        return []
    
    # Get all directories that seem to be experiment directories
    exp_dirs = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path):
            # Check if it contains a config.yaml file (indicating it's an experiment dir)
            if os.path.exists(os.path.join(full_path, 'config.yaml')):
                exp_dirs.append(item)
    
    return exp_dirs

def extract_key_config_info(config: Union[Dict[str, Any], DictConfig]) -> Dict[str, Any]:
    """
    Extract key information from a configuration that's useful for experiment listings.
    """
    result = {}
    
    # Extract encoder info
    if 'encoder' in config:
        if isinstance(config['encoder'], dict) or hasattr(config['encoder'], 'keys'):
            encoder_type = config['encoder'].get('_target_', 'Unknown')
            # Get just the class name from the target path
            if '.' in encoder_type:
                encoder_type = encoder_type.split('.')[-1]
            result['encoder'] = encoder_type
        else:
            result['encoder'] = str(config['encoder'])
    else:
        result['encoder'] = 'N/A'
    
    # Extract generator info
    if 'generator' in config:
        if isinstance(config['generator'], dict) or hasattr(config['generator'], 'keys'):
            generator_type = config['generator'].get('_target_', 'Unknown')
            # Get just the class name from the target path
            if '.' in generator_type:
                generator_type = generator_type.split('.')[-1]
            result['generator'] = generator_type
        else:
            result['generator'] = str(config['generator'])
    else:
        result['generator'] = 'N/A'
    
    # Extract model info
    if 'model' in config:
        if isinstance(config['model'], dict) or hasattr(config['model'], 'keys'):
            model_type = config['model'].get('_target_', 'Unknown')
            # Get just the class name from the target path
            if '.' in model_type:
                model_type = model_type.split('.')[-1]
            result['model'] = model_type
        else:
            result['model'] = str(config['model'])
    else:
        result['model'] = 'N/A'
    
    # Extract dataset info
    if 'dataset' in config:
        if isinstance(config['dataset'], dict) or hasattr(config['dataset'], 'keys'):
            dataset_type = config['dataset'].get('name', config['dataset'].get('_target_', 'Unknown'))
            # Get just the dataset name
            if '.' in dataset_type:
                dataset_type = dataset_type.split('.')[-1]
            result['dataset'] = dataset_type
        else:
            result['dataset'] = str(config['dataset'])
    else:
        result['dataset'] = 'N/A'
    
    # Extract latent dimension
    if 'latent_dim' in config:
        result['latent_dim'] = config['latent_dim']
    elif 'model' in config and isinstance(config['model'], dict) and 'latent_dim' in config['model']:
        result['latent_dim'] = config['model']['latent_dim']
    elif 'encoder' in config and isinstance(config['encoder'], dict) and 'latent_dim' in config['encoder']:
        result['latent_dim'] = config['encoder']['latent_dim']
    else:
        result['latent_dim'] = 'N/A'
    
    # Extract training info
    if 'trainer' in config and isinstance(config['trainer'], dict):
        result['training_epochs'] = config['trainer'].get('max_epochs', 'N/A')
    elif 'training' in config and isinstance(config['training'], dict):
        result['training_epochs'] = config['training'].get('max_epochs', 'N/A')
    else:
        result['training_epochs'] = 'N/A'
    
    # Extract batch size
    if 'datamodule' in config and isinstance(config['datamodule'], dict):
        result['batch_size'] = config['datamodule'].get('batch_size', 'N/A')
    elif 'data' in config and isinstance(config['data'], dict):
        result['batch_size'] = config['data'].get('batch_size', 'N/A')
    else:
        result['batch_size'] = 'N/A'
    
    return result

def get_experiment_info(experiment_dir: str, load_checkpoints: bool = False, force_no_checkpoints: bool = False) -> Dict[str, Any]:
    """
    Get information about an experiment.
    
    Args:
        experiment_dir: The directory of the experiment
        load_checkpoints: Whether to load checkpoint information (may be slow)
        force_no_checkpoints: If True, guarantees that checkpoints won't be loaded regardless of other settings
        
    Returns:
        A dictionary with experiment information
    """
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory {experiment_dir} does not exist")
    
    result = {
        'name': os.path.basename(experiment_dir),
        'dir': experiment_dir,
    }
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'config.yaml')
    if os.path.exists(config_path):
        try:
            config = OmegaConf.load(config_path)
            result['config'] = config
            result['config_hash'] = hash_config(config)
            
            # Extract key config information
            key_info = extract_key_config_info(config)
            result.update(key_info)
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            result['config'] = None
            result['config_hash'] = None
    else:
        logger.warning(f"No config.yaml found in {experiment_dir}")
        result['config'] = None
        result['config_hash'] = None
    
    # Find checkpoints
    checkpoints = glob.glob(os.path.join(experiment_dir, "*.ckpt"))
    result['checkpoints'] = sorted(checkpoints)
    
    # Check if best model exists
    best_model_path = os.path.join(experiment_dir, "best_model.ckpt")
    result['best_model'] = best_model_path if os.path.exists(best_model_path) else None
    
    # Try to load best loss and epoch from metrics.json first to avoid loading checkpoints
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            result['best_loss'] = metrics.get('best_loss')
            result['best_epoch'] = metrics.get('best_epoch')
            
            # If metrics were loaded successfully, no need to load checkpoints
            load_checkpoints = False
        except Exception as e:
            logger.error(f"Error loading metrics from {metrics_path}: {e}")
    
    # Try to load best model if available and checkpoints should be loaded
    if load_checkpoints and not force_no_checkpoints and result['best_model']:
        try:
            ckpt = torch.load(result['best_model'], map_location=torch.device('cpu'))
            result['best_loss'] = ckpt.get('best_score', None)
            result['best_epoch'] = ckpt.get('epoch', None)
        except Exception as e:
            logger.error(f"Error loading best model from {result['best_model']}: {e}")
            result['best_loss'] = None
            result['best_epoch'] = None
    elif force_no_checkpoints or not load_checkpoints:
        # If forcing no checkpoints or not loading them, make sure they're not set from before
        if 'best_loss' not in result:
            result['best_loss'] = 'N/A'
        if 'best_epoch' not in result:
            result['best_epoch'] = 'N/A'
    
    return result

def get_all_experiments_info(base_dir: str, load_checkpoints: bool = False, sort_by: str = None) -> List[Dict[str, Any]]:
    """
    Get information about all experiments in a batch-efficient manner.
    
    Args:
        base_dir: The base directory containing experiment directories
        load_checkpoints: Whether to load checkpoint information (may be slow)
        sort_by: Field to sort the results by
        
    Returns:
        A list of dictionaries with experiment information
    """
    experiments = find_all_experiments(base_dir)
    
    if not experiments:
        return []
    
    # Collect information for all experiments
    experiments_info = []
    for exp_name in experiments:
        exp_dir = os.path.join(base_dir, exp_name)
        try:
            # Force no checkpoints for listing by default
            info = get_experiment_info(exp_dir, load_checkpoints=load_checkpoints, force_no_checkpoints=not load_checkpoints)
            experiments_info.append(info)
        except Exception as e:
            logger.error(f"Error getting info for {exp_name}: {e}")
    
    # Sort results if requested
    if sort_by and experiments_info:
        # Try numeric sorting first
        try:
            # For numeric fields like best_loss
            experiments_info.sort(key=lambda x: (
                float('inf') if x.get(sort_by) is None or x.get(sort_by) == 'N/A' 
                else float(x.get(sort_by))
            ))
        except (ValueError, TypeError):
            # Fall back to string sorting for non-numeric fields like name
            experiments_info.sort(key=lambda x: str(x.get(sort_by, '')))
    
    return experiments_info

def find_experiments_by_config(config: Union[Dict[str, Any], DictConfig], base_dir: str = './outputs', exact_match: bool = True) -> List[str]:
    """
    Find experiments that match a given configuration.
    
    Args:
        config: The configuration to match
        base_dir: The base directory containing experiment directories
        exact_match: Whether to require an exact match or just partial matching
        
    Returns:
        A list of experiment directories that match the configuration
    """
    config_hash = hash_config(config)
    matching_experiments = []
    
    for exp_name in find_all_experiments(base_dir):
        exp_dir = os.path.join(base_dir, exp_name)
        config_path = os.path.join(exp_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            continue
        
        try:
            exp_config = OmegaConf.load(config_path)
            exp_hash = hash_config(exp_config)
            
            if exact_match and exp_hash == config_hash:
                matching_experiments.append(exp_dir)
            elif not exact_match:
                # Partial matching - check if all keys in config exist in exp_config with the same values
                match = True
                for key, value in config.items():
                    if key not in exp_config or exp_config[key] != value:
                        match = False
                        break
                
                if match:
                    matching_experiments.append(exp_dir)
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    return matching_experiments

def load_best_model(experiment_dir: str):
    """
    Load the best model from an experiment.
    
    Args:
        experiment_dir: The directory of the experiment
        
    Returns:
        The loaded model checkpoint
    """
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory {experiment_dir} does not exist")
    
    best_model_path = os.path.join(experiment_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise ValueError(f"No best model found in {experiment_dir}")
    
    return torch.load(best_model_path, map_location=torch.device('cpu'), weights_only=False)

def compare_experiments(exp_dir1: str, exp_dir2: str) -> Dict[str, Any]:
    """
    Compare two experiments and return their differences.
    
    Args:
        exp_dir1: The directory of the first experiment
        exp_dir2: The directory of the second experiment
        
    Returns:
        A dictionary with comparison results
    """
    if not os.path.exists(exp_dir1):
        return {'error': f"Experiment directory {exp_dir1} does not exist"}
    
    if not os.path.exists(exp_dir2):
        return {'error': f"Experiment directory {exp_dir2} does not exist"}
    
    result = {
        'exp1': os.path.basename(exp_dir1),
        'exp2': os.path.basename(exp_dir2)
    }
    
    # Get information without loading checkpoints
    info1 = get_experiment_info(exp_dir1, force_no_checkpoints=True)
    info2 = get_experiment_info(exp_dir2, force_no_checkpoints=True)
    
    # Compare configurations
    if info1['config'] is not None and info2['config'] is not None:
        config1 = OmegaConf.to_container(info1['config'], resolve=True) if not isinstance(info1['config'], dict) else info1['config']
        config2 = OmegaConf.to_container(info2['config'], resolve=True) if not isinstance(info2['config'], dict) else info2['config']
        
        config_diff = {}
        
        # Find keys in both configs
        all_keys = set(config1.keys()).union(set(config2.keys()))
        
        for key in all_keys:
            val1 = config1.get(key, None)
            val2 = config2.get(key, None)
            
            if val1 != val2:
                config_diff[key] = (val1, val2)
        
        result['config_differences'] = config_diff
    
    # Compare performance metrics if available
    if 'best_loss' in info1 and 'best_loss' in info2:
        perf_diff = {}
        
        if info1['best_loss'] is not None and info2['best_loss'] is not None and info1['best_loss'] != 'N/A' and info2['best_loss'] != 'N/A':
            perf_diff['best_loss'] = (info1['best_loss'], info2['best_loss'])
        
        if info1['best_epoch'] is not None and info2['best_epoch'] is not None and info1['best_epoch'] != 'N/A' and info2['best_epoch'] != 'N/A':
            perf_diff['best_epoch'] = (info1['best_epoch'], info2['best_epoch'])
        
        result['performance_differences'] = perf_diff
    
    return result

def create_metrics_file(experiment_dir: str, metrics: Dict[str, Any]) -> bool:
    """
    Create a metrics.json file in the experiment directory.
    This helps avoid loading checkpoints when retrieving metrics.
    
    Args:
        experiment_dir: The directory of the experiment
        metrics: The metrics to save
        
    Returns:
        Whether the operation was successful
    """
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return False
    
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error creating metrics file {metrics_path}: {e}")
        return False 