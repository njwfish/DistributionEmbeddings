#!/usr/bin/env python
import argparse
import os
import logging
import yaml
import json
import sys
from utils.hash_utils import hash_config, find_matching_output_dir
from utils.experiment_utils import (
    find_all_experiments, 
    get_experiment_info, 
    find_experiments_by_config,
    compare_experiments,
    load_best_model
)
from omegaconf import OmegaConf
from tabulate import tabulate
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_experiments(args):
    """List all experiments and their key metrics."""
    exps = find_all_experiments(args.output_dir)
    
    if not exps:
        logger.info(f"No experiments found in {args.output_dir}")
        return
    
    # Collect experiment information
    experiments_info = []
    for exp_name in exps:
        exp_dir = os.path.join(args.output_dir, exp_name)
        try:
            info = get_experiment_info(exp_dir)
            experiments_info.append({
                'name': info['name'],
                'hash': info['config_hash'],
                'best_loss': info.get('best_loss', 'N/A'),
                'best_epoch': info.get('best_epoch', 'N/A'),
                'checkpoints': len(info['checkpoints']),
                'has_best_model': 'Yes' if info['best_model'] else 'No'
            })
        except Exception as e:
            logger.error(f"Error processing {exp_name}: {e}")
    
    # Sort by best loss if available
    experiments_info.sort(key=lambda x: 
                         float(x['best_loss']) if x['best_loss'] != 'N/A' else float('inf'))
    
    # Display as a table
    headers = ['Name', 'Config Hash', 'Best Loss', 'Best Epoch', 'Checkpoints', 'Best Model']
    table_data = [
        [exp['name'], exp['hash'], exp['best_loss'], exp['best_epoch'], 
         exp['checkpoints'], exp['has_best_model']]
        for exp in experiments_info
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"Total experiments: {len(experiments_info)}")

def show_experiment(args):
    """Show detailed information about a specific experiment."""
    # Find experiment by name or hash
    exp_dir = None
    if os.path.isdir(os.path.join(args.output_dir, args.experiment)):
        exp_dir = os.path.join(args.output_dir, args.experiment)
    else:
        # Try to find by hash
        for exp_name in find_all_experiments(args.output_dir):
            if args.experiment in exp_name:
                exp_dir = os.path.join(args.output_dir, exp_name)
                break
    
    if exp_dir is None:
        logger.error(f"Experiment '{args.experiment}' not found")
        return
    
    try:
        info = get_experiment_info(exp_dir)
        
        print(f"\n=== Experiment: {info['name']} ===")
        print(f"Directory: {info['dir']}")
        print(f"Config Hash: {info['config_hash']}")
        
        if info.get('best_loss') is not None:
            print(f"Best Loss: {info['best_loss']}")
        if info.get('best_epoch') is not None:
            print(f"Best Epoch: {info['best_epoch']}")
            
        print(f"Checkpoints: {len(info['checkpoints'])}")
        if info['checkpoints']:
            for checkpoint in info['checkpoints'][-5:]:  # Show only the last 5 checkpoints
                print(f"  - {os.path.basename(checkpoint)}")
            if len(info['checkpoints']) > 5:
                print(f"  ... and {len(info['checkpoints']) - 5} more")
        
        # Display configuration if available
        if info['config'] is not None:
            print("\n--- Configuration ---")
            if args.format == 'yaml':
                if isinstance(info['config'], dict):
                    print(yaml.dump(info['config'], default_flow_style=False))
                else:
                    print(OmegaConf.to_yaml(info['config']))
            else:
                if isinstance(info['config'], dict):
                    print(json.dumps(info['config'], indent=2))
                else:
                    print(json.dumps(OmegaConf.to_container(info['config'], resolve=True), indent=2))
    
    except Exception as e:
        logger.error(f"Error showing experiment: {e}")

def compare_exps(args):
    """Compare two experiments."""
    exp1 = args.experiment1
    exp2 = args.experiment2
    
    # Find experiment directories
    exp1_dir = os.path.join(args.output_dir, exp1) if os.path.isdir(os.path.join(args.output_dir, exp1)) else None
    exp2_dir = os.path.join(args.output_dir, exp2) if os.path.isdir(os.path.join(args.output_dir, exp2)) else None
    
    # Try to find by hash
    if exp1_dir is None:
        for exp_name in find_all_experiments(args.output_dir):
            if exp1 in exp_name:
                exp1_dir = os.path.join(args.output_dir, exp_name)
                break
    
    if exp2_dir is None:
        for exp_name in find_all_experiments(args.output_dir):
            if exp2 in exp_name:
                exp2_dir = os.path.join(args.output_dir, exp_name)
                break
    
    if exp1_dir is None or exp2_dir is None:
        if exp1_dir is None:
            logger.error(f"Experiment '{exp1}' not found")
        if exp2_dir is None:
            logger.error(f"Experiment '{exp2}' not found")
        return
    
    try:
        comparison = compare_experiments(exp1_dir, exp2_dir)
        
        print(f"\n=== Comparing Experiments ===")
        print(f"Experiment 1: {comparison['exp1']}")
        print(f"Experiment 2: {comparison['exp2']}")
        
        # Show performance differences
        if comparison.get('performance_differences'):
            print("\n--- Performance Differences ---")
            for metric, (val1, val2) in comparison['performance_differences'].items():
                print(f"{metric}: {val1} vs {val2}")
                # Calculate percentage difference if possible
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val1 != 0:
                    pct_diff = ((val2 - val1) / abs(val1)) * 100
                    print(f"  Difference: {pct_diff:.2f}%")
        
        # Show config differences
        if comparison.get('config_differences'):
            print("\n--- Configuration Differences ---")
            for key, (val1, val2) in comparison['config_differences'].items():
                print(f"{key}: {val1} → {val2}")
        
        if 'error' in comparison:
            print(f"\nError: {comparison['error']}")
    
    except Exception as e:
        logger.error(f"Error comparing experiments: {e}")

def find_exp_by_config(args):
    """Find experiments matching a configuration file."""
    # Load configuration file
    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        return
    
    try:
        if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
            config = OmegaConf.load(args.config_file)
        else:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
        
        # Find matching experiments
        matching_exps = find_experiments_by_config(
            config, 
            base_dir=args.output_dir,
            exact_match=not args.partial
        )
        
        if not matching_exps:
            print(f"No matching experiments found for the given configuration")
            return
        
        print(f"\n=== Matching Experiments ===")
        for exp_dir in matching_exps:
            info = get_experiment_info(exp_dir)
            print(f"- {info['name']}")
            if info.get('best_loss') is not None:
                print(f"  Best Loss: {info['best_loss']}")
            if info.get('best_epoch') is not None:
                print(f"  Best Epoch: {info['best_epoch']}")
            print(f"  Directory: {info['dir']}")
            print("")
    
    except Exception as e:
        logger.error(f"Error finding experiments by config: {e}")

def compute_hash(args):
    """Compute and display the hash for a configuration file."""
    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        return
    
    try:
        if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
            config = OmegaConf.load(args.config_file)
        else:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
        
        config_hash = hash_config(config)
        print(f"Configuration hash: {config_hash}")
        
        # Check if an experiment with this hash exists
        existing_dir = find_matching_output_dir(config, base_dir=args.output_dir)
        if existing_dir:
            print(f"Found existing experiment: {os.path.basename(existing_dir)}")
            print(f"Directory: {existing_dir}")
    
    except Exception as e:
        logger.error(f"Error computing hash: {e}")

def main():
    parser = argparse.ArgumentParser(description='Experiment Management CLI')
    parser.add_argument('--output-dir', default='./outputs', help='Base directory for experiment outputs')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show details of an experiment')
    show_parser.add_argument('experiment', help='Experiment name or hash')
    show_parser.add_argument('--format', choices=['json', 'yaml'], default='yaml', help='Output format for config')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two experiments')
    compare_parser.add_argument('experiment1', help='First experiment name or hash')
    compare_parser.add_argument('experiment2', help='Second experiment name or hash')
    
    # Find command
    find_parser = subparsers.add_parser('find', help='Find experiments matching a config')
    find_parser.add_argument('config_file', help='Path to config file')
    find_parser.add_argument('--partial', action='store_true', help='Allow partial matches')
    
    # Hash command
    hash_parser = subparsers.add_parser('hash', help='Compute hash for a config file')
    hash_parser.add_argument('config_file', help='Path to config file')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_experiments(args)
    elif args.command == 'show':
        show_experiment(args)
    elif args.command == 'compare':
        compare_exps(args)
    elif args.command == 'find':
        find_exp_by_config(args)
    elif args.command == 'hash':
        compute_hash(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 