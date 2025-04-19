#!/usr/bin/env python
import argparse
import os
import logging
import yaml
import json
import sys
import shutil
from utils.hash_utils import hash_config, find_matching_output_dir
from utils.experiment_utils import (
    find_all_experiments, 
    get_experiment_info,
    get_all_experiments_info,
    find_experiments_by_config,
    compare_experiments,
    load_best_model,
    create_metrics_file
)
from omegaconf import OmegaConf
from tabulate import tabulate
from typing import Dict, Any, List
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_experiments(args):
    """List all experiments and their key metrics."""
    start_time = time.time()
    logger.info(f"Loading experiments from {args.output_dir}...")
    
    # Use the efficient batch method - never load checkpoints for listing
    experiments_info = get_all_experiments_info(
        args.output_dir, 
        load_checkpoints=False,  # Never load checkpoints for listing
        sort_by=args.sort_by
    )
    
    if not experiments_info:
        logger.info(f"No experiments found in {args.output_dir}")
        return
    
    # Determine columns to display (default plus requested columns)
    default_columns = ['name', 'hash']
    metric_columns = ['best_loss', 'best_epoch']
    model_columns = ['encoder', 'generator', 'model', 'dataset']
    training_columns = ['latent_dim', 'training_epochs', 'batch_size', 'checkpoints']
    
    # By default, show more useful information including model components
    shown_columns = ['name', 'hash', 'encoder', 'generator', 'model', 'dataset']
    
    # Override with user selections if specified
    if args.show_metrics:
        shown_columns.extend([col for col in metric_columns if col not in shown_columns])
    
    if args.show_models:
        shown_columns.extend([col for col in model_columns if col not in shown_columns])
    
    if args.show_training:
        shown_columns.extend([col for col in training_columns if col not in shown_columns])
    
    if args.columns:
        # User specified custom columns
        custom_columns = [col.strip() for col in args.columns.split(',')]
        # Filter to only include valid columns
        valid_columns = default_columns + metric_columns + model_columns + training_columns
        custom_columns = [col for col in custom_columns if col in valid_columns]
        shown_columns = custom_columns
    
    # Prepare data for the table
    table_data = []
    for exp in experiments_info:
        row = []
        for col in shown_columns:
            if col == 'name':
                # Remove hash from name display for cleaner output
                name_parts = exp['name'].split('_')
                if len(name_parts) > 1 and len(name_parts[-1]) > 8:  # Likely a hash
                    name = '_'.join(name_parts[:-1])
                else:
                    name = exp['name']
                row.append(name)
            elif col == 'hash':
                if exp.get('config_hash') and len(exp.get('config_hash', '')) > 8:
                    row.append(exp.get('config_hash', 'N/A'))  # Show truncated hash
                else:
                    row.append(exp.get('config_hash', 'N/A'))
            elif col == 'checkpoints':
                row.append(len(exp.get('checkpoints', [])))
            else:
                row.append(exp.get(col, 'N/A'))
        table_data.append(row)
    
    # Display as a table
    headers = [col.replace('_', ' ').title() for col in shown_columns]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"Total experiments: {len(experiments_info)}")
    
    # Report loading time
    end_time = time.time()
    logger.info(f"Listed {len(experiments_info)} experiments in {end_time - start_time:.2f} seconds")

def export_experiments(args):
    """Export experiment information to CSV."""
    start_time = time.time()
    logger.info(f"Loading experiments from {args.output_dir}...")
    
    # Use the efficient batch method - never load checkpoints for exporting
    experiments_info = get_all_experiments_info(
        args.output_dir, 
        load_checkpoints=False,
        sort_by=args.sort_by
    )
    
    if not experiments_info:
        logger.info(f"No experiments found in {args.output_dir}")
        return
    
    # Determine all possible columns
    all_columns = set()
    for exp in experiments_info:
        all_columns.update(exp.keys())
    
    # Remove columns that aren't useful for CSV
    columns_to_remove = ['dir', 'config', 'best_model', 'checkpoints']
    all_columns = [col for col in all_columns if col not in columns_to_remove]
    
    # Prepare CSV data
    csv_data = []
    csv_data.append(','.join(all_columns))  # Header row
    
    for exp in experiments_info:
        row = []
        for col in all_columns:
            value = exp.get(col, '')
            # Handle values that might contain commas
            if isinstance(value, str) and ',' in value:
                value = f'"{value}"'
            elif value is None:
                value = ''
            row.append(str(value))
        csv_data.append(','.join(row))
    
    # Write to file
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(csv_data))
    
    logger.info(f"Exported {len(experiments_info)} experiments to {args.output_file}")
    
    # Report time
    end_time = time.time()
    logger.info(f"Export completed in {end_time - start_time:.2f} seconds")

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
        start_time = time.time()
        info = get_experiment_info(exp_dir, load_checkpoints=args.load_checkpoints, force_no_checkpoints=not args.load_checkpoints)
        
        print(f"\n=== Experiment: {info['name']} ===")
        print(f"Directory: {info['dir']}")
        print(f"Config Hash: {info['config_hash']}")
        
        # Get the actual config for extracting more detailed information
        config = info.get('config')
        
        # Display key model components with direct config access
        print("\n--- Model Components ---")
        
        # Display encoder details
        print("Encoder:")
        if config and 'encoder' in config:
            if isinstance(config['encoder'], dict) or hasattr(config['encoder'], 'keys'):
                target = config['encoder'].get('_target_', 'Not specified')
                print(f"  Type: {target.split('.')[-1] if '.' in target else target}")
                
                # Show important encoder parameters
                for key in ['latent_dim', 'hidden_dims', 'activation', 'dropout']:
                    if key in config['encoder']:
                        print(f"  {key.replace('_', ' ').title()}: {config['encoder'][key]}")
            else:
                print(f"  {config['encoder']}")
        else:
            print(f"  {info.get('encoder', 'Not specified')}")
        
        # Display generator details
        print("\nGenerator:")
        if config and 'generator' in config:
            if isinstance(config['generator'], dict) or hasattr(config['generator'], 'keys'):
                target = config['generator'].get('_target_', 'Not specified')
                print(f"  Type: {target.split('.')[-1] if '.' in target else target}")
                
                # Show important generator parameters
                for key in ['noise_steps', 'beta_schedule', 'beta_start', 'beta_end']:
                    if key in config['generator']:
                        print(f"  {key.replace('_', ' ').title()}: {config['generator'][key]}")
            else:
                print(f"  {config['generator']}")
        else:
            print(f"  {info.get('generator', 'Not specified')}")
        
        # Display model details
        print("\nModel:")
        if config and 'model' in config:
            if isinstance(config['model'], dict) or hasattr(config['model'], 'keys'):
                target = config['model'].get('_target_', 'Not specified')
                print(f"  Type: {target.split('.')[-1] if '.' in target else target}")
                
                # Show important model parameters
                for key in ['hidden_dims', 'latent_dim', 'num_layers', 'activation']:
                    if key in config['model']:
                        print(f"  {key.replace('_', ' ').title()}: {config['model'][key]}")
            else:
                print(f"  {config['model']}")
        else:
            print(f"  {info.get('model', 'Not specified')}")
        
        # Display dataset details
        print("\nDataset:")
        if config and 'dataset' in config:
            if isinstance(config['dataset'], dict) or hasattr(config['dataset'], 'keys'):
                name = config['dataset'].get('name', 
                                            config['dataset'].get('_target_', 'Not specified'))
                print(f"  Type: {name.split('.')[-1] if '.' in name else name}")
                
                # Show important dataset parameters
                for key in ['batch_size', 'num_samples', 'set_size', 'dim']:
                    if key in config['dataset']:
                        print(f"  {key.replace('_', ' ').title()}: {config['dataset'][key]}")
            else:
                print(f"  {config['dataset']}")
        else:
            print(f"  {info.get('dataset', 'Not specified')}")
        
        # Display training info
        print("\n--- Training Info ---")
        if config and ('trainer' in config or 'training' in config):
            trainer_config = config.get('trainer', config.get('training', {}))
            if isinstance(trainer_config, dict) or hasattr(trainer_config, 'keys'):
                for key in ['max_epochs', 'lr', 'batch_size', 'optimizer', 'scheduler']:
                    if key in trainer_config:
                        print(f"{key.replace('_', ' ').title()}: {trainer_config[key]}")
        else:
            print(f"Training Epochs: {info.get('training_epochs', 'Not specified')}")
            print(f"Batch Size: {info.get('batch_size', 'Not specified')}")
        
        print(f"Checkpoints: {len(info['checkpoints'])}")
        
        # Display metrics
        print("\n--- Metrics ---")
        if info.get('best_loss') is not None and info.get('best_loss') != 'N/A':
            print(f"Best Loss: {info['best_loss']}")
        else:
            print("Best Loss: Not available (use --load-checkpoints to attempt loading)")
            
        if info.get('best_epoch') is not None and info.get('best_epoch') != 'N/A':
            print(f"Best Epoch: {info['best_epoch']}")
        else:
            print("Best Epoch: Not available (use --load-checkpoints to attempt loading)")
        
        # Display checkpoints
        if info['checkpoints']:
            print("\n--- Latest Checkpoints ---")
            for checkpoint in info['checkpoints'][-5:]:  # Show only the last 5 checkpoints
                print(f"  - {os.path.basename(checkpoint)}")
            if len(info['checkpoints']) > 5:
                print(f"  ... and {len(info['checkpoints']) - 5} more")
        
        # Display configuration if requested
        if args.show_config:
            print("\n--- Full Configuration ---")
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
        
        end_time = time.time()
        logger.info(f"Loaded experiment details in {end_time - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error showing experiment: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

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
        start_time = time.time()
        comparison = compare_experiments(exp1_dir, exp2_dir)
        
        print(f"\n=== Comparing Experiments ===")
        print(f"Experiment 1: {comparison['exp1']}")
        print(f"Experiment 2: {comparison['exp2']}")
        
        # Get experiment info for both
        exp1_info = get_experiment_info(exp1_dir, force_no_checkpoints=True)
        exp2_info = get_experiment_info(exp2_dir, force_no_checkpoints=True)
        
        # Show model component differences
        model_components = ['encoder', 'generator', 'model', 'dataset', 'training_epochs', 'batch_size', 'latent_dim']
        model_diffs = {}
        for comp in model_components:
            val1 = exp1_info.get(comp, 'N/A')
            val2 = exp2_info.get(comp, 'N/A')
            if val1 != val2:
                model_diffs[comp] = (val1, val2)
        
        if model_diffs:
            print("\n--- Model Component Differences ---")
            for comp, (val1, val2) in model_diffs.items():
                print(f"{comp.replace('_', ' ').title()}: {val1} → {val2}")
        
        # Show performance differences
        if comparison.get('performance_differences'):
            print("\n--- Performance Differences ---")
            for metric, (val1, val2) in comparison['performance_differences'].items():
                print(f"{metric}: {val1} vs {val2}")
                # Calculate percentage difference if possible
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val1 != 0:
                    pct_diff = ((val2 - val1) / abs(val1)) * 100
                    print(f"  Difference: {pct_diff:.2f}%")
        
        # Show config differences (only if detailed)
        if comparison.get('config_differences') and args.detailed:
            print("\n--- Detailed Configuration Differences ---")
            for key, (val1, val2) in comparison['config_differences'].items():
                print(f"{key}: {val1} → {val2}")
        
        if 'error' in comparison:
            print(f"\nError: {comparison['error']}")
        
        end_time = time.time()
        logger.info(f"Compared experiments in {end_time - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error comparing experiments: {e}")

def find_exp_by_config(args):
    """Find experiments matching a configuration file."""
    # Load configuration file
    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        return
    
    try:
        start_time = time.time()
        
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
            exp_info = get_experiment_info(exp_dir, force_no_checkpoints=True)
            print(f"\n- {exp_info['name']}")
            print(f"  Directory: {exp_info['dir']}")
            print(f"  Encoder: {exp_info['encoder']}")
            print(f"  Generator: {exp_info['generator']}")
            print(f"  Dataset: {exp_info['dataset']}")
            print(f"  Latent Dim: {exp_info['latent_dim']}")
            if exp_info.get('best_loss') is not None and exp_info.get('best_loss') != 'N/A':
                print(f"  Best Loss: {exp_info['best_loss']}")
        
        print(f"\nFound {len(matching_exps)} matching experiment(s)")
        
        end_time = time.time()
        logger.info(f"Found experiments in {end_time - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error finding experiments: {e}")

def hash_config_file(args):
    """Calculate the hash for a config file."""
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
        print(f"Config hash: {config_hash}")
        
        # Check if experiment exists
        matching_dir = find_matching_output_dir(config, args.output_dir)
        if matching_dir:
            print(f"Matching experiment found: {os.path.basename(matching_dir)}")
        else:
            print("No matching experiment found")
    
    except Exception as e:
        logger.error(f"Error calculating hash: {e}")

def create_metrics_files(args):
    """Create metrics.json files for experiments to avoid loading checkpoints."""
    exps = find_all_experiments(args.output_dir)
    
    if not exps:
        logger.info(f"No experiments found in {args.output_dir}")
        return
    
    created_count = 0
    skipped_count = 0
    
    for exp_name in exps:
        exp_dir = os.path.join(args.output_dir, exp_name)
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        
        # Skip if metrics file already exists
        if os.path.exists(metrics_path) and not args.force:
            skipped_count += 1
            continue
        
        # Get info by loading checkpoint
        try:
            info = get_experiment_info(exp_dir, load_checkpoints=True, force_no_checkpoints=False)
            
            # Extract relevant metrics
            metrics = {
                'best_epoch': info.get('best_epoch'),
                'best_loss': info.get('best_loss')
            }
            
            # Only create metrics file if we found valid metrics
            if metrics['best_epoch'] is not None or metrics['best_loss'] is not None:
                if create_metrics_file(exp_dir, metrics):
                    created_count += 1
                    logger.info(f"Created metrics file for {exp_name}")
            else:
                logger.warning(f"No metrics found for {exp_name}")
        except Exception as e:
            logger.error(f"Error processing {exp_name}: {e}")
    
    logger.info(f"Created {created_count} metrics files, skipped {skipped_count} existing files")

def cleanup_dead_experiments(args):
    """Remove 'dead' experiments that only have a config file and no outputs."""
    exps = find_all_experiments(args.output_dir)
    
    if not exps:
        logger.info(f"No experiments found in {args.output_dir}")
        return
    
    removed_count = 0
    preserved_count = 0
    empty_exps = []
    
    for exp_name in exps:
        exp_dir = os.path.join(args.output_dir, exp_name)
        
        # Count files other than config.yaml
        files = os.listdir(exp_dir)
        non_config_files = [f for f in files if f != 'config.yaml']
        
        # Check for checkpoints or any other output files
        has_checkpoints = any(f.endswith('.ckpt') for f in files)
        has_metrics = 'metrics.json' in files
        has_outputs = has_checkpoints or has_metrics or len(non_config_files) > 0
        
        # Determine if this is a "dead" experiment (only has config.yaml)
        is_dead = not has_outputs
        
        # If it's dead and we're not keeping all experiments, add to removal list
        if is_dead and not args.keep_all:
            empty_exps.append((exp_name, exp_dir))
        else:
            preserved_count += 1
    
    # If we found experiments to remove
    if empty_exps:
        # Print experiments that will be removed
        print(f"\n=== Dead Experiments ===")
        print(f"The following {len(empty_exps)} experiments will be removed:")
        for exp_name, _ in empty_exps:
            print(f"- {exp_name}")
        
        if args.dry_run:
            print(f"\nDRY RUN - No experiments were actually removed")
            print(f"Found {len(empty_exps)} dead experiments and {preserved_count} active experiments")
            return
        
        # Ask for confirmation if not forced
        if not args.force:
            confirmation = input(f"\nAre you sure you want to remove these {len(empty_exps)} experiments? (y/N): ")
            if confirmation.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                return
        
        # Remove the experiments
        for exp_name, exp_dir in empty_exps:
            try:
                if args.move_to:
                    # Create archive directory if it doesn't exist
                    archive_dir = os.path.join(args.move_to, "dead_experiments")
                    os.makedirs(archive_dir, exist_ok=True)
                    
                    # Move to archive instead of deleting
                    dest_path = os.path.join(archive_dir, exp_name)
                    shutil.move(exp_dir, dest_path)
                    logger.info(f"Moved {exp_name} to {dest_path}")
                else:
                    # Delete the experiment directory
                    shutil.rmtree(exp_dir)
                    logger.info(f"Removed {exp_name}")
                
                removed_count += 1
            except Exception as e:
                logger.error(f"Error removing {exp_name}: {e}")
    
        # Report results
        if args.move_to:
            logger.info(f"Moved {removed_count} dead experiments to {os.path.join(args.move_to, 'dead_experiments')}")
        else:
            logger.info(f"Removed {removed_count} dead experiments")
        logger.info(f"Preserved {preserved_count} active experiments")
    else:
        logger.info("No dead experiments found to remove")
        if preserved_count > 0:
            logger.info(f"All {preserved_count} experiments have outputs")

def main():
    parser = argparse.ArgumentParser(description='Distribution Embeddings Experiment CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Global arguments
    parser.add_argument('--output-dir', '-o', default='./outputs', help='Output directory')
    
    # List experiments command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.add_argument('--sort-by', default='name', 
                            help='Field to sort results by (name, encoder, generator, model, dataset, best_loss)')
    list_parser.add_argument('--show-metrics', action='store_true', help='Show metrics columns')
    list_parser.add_argument('--show-models', action='store_true', help='Show all model component columns')
    list_parser.add_argument('--show-training', action='store_true', help='Show training info columns')
    list_parser.add_argument('--columns', help='Comma-separated list of columns to display')
    
    # Export experiments command
    export_parser = subparsers.add_parser('export', help='Export experiments to CSV')
    export_parser.add_argument('--output-file', '-f', required=True, help='Output CSV file')
    export_parser.add_argument('--sort-by', default='name', help='Field to sort results by')
    
    # Show experiment command
    show_parser = subparsers.add_parser('show', help='Show details for a specific experiment')
    show_parser.add_argument('experiment', help='Experiment name or hash')
    show_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml', help='Config display format')
    show_parser.add_argument('--load-checkpoints', action='store_true', 
                            help='Load checkpoints for metrics (slower but may provide additional info)')
    show_parser.add_argument('--show-config', action='store_true', help='Show full configuration')
    show_parser.add_argument('--debug', action='store_true', help='Show detailed error information if an error occurs')
    
    # Compare experiments command
    compare_parser = subparsers.add_parser('compare', help='Compare two experiments')
    compare_parser.add_argument('experiment1', help='First experiment name or hash')
    compare_parser.add_argument('experiment2', help='Second experiment name or hash')
    compare_parser.add_argument('--detailed', action='store_true', help='Show detailed config differences')
    
    # Find experiments command
    find_parser = subparsers.add_parser('find', help='Find experiments matching a config file')
    find_parser.add_argument('config_file', help='Config file to match')
    find_parser.add_argument('--partial', action='store_true', help='Allow partial matches')
    
    # Hash config command
    hash_parser = subparsers.add_parser('hash', help='Calculate hash for a config file')
    hash_parser.add_argument('config_file', help='Config file to hash')
    
    # Create metrics files command
    metrics_parser = subparsers.add_parser('create-metrics', 
                                        help='Create metrics.json files for experiments to avoid loading checkpoints')
    metrics_parser.add_argument('--force', action='store_true', help='Overwrite existing metrics files')
    
    # Cleanup dead experiments command
    cleanup_parser = subparsers.add_parser('cleanup', 
                                        help='Remove experiments that only have a config file and no outputs')
    cleanup_parser.add_argument('--force', action='store_true', 
                              help='Remove without asking for confirmation')
    cleanup_parser.add_argument('--dry-run', action='store_true', 
                              help='Show what would be removed without actually removing')
    cleanup_parser.add_argument('--move-to', 
                              help='Move dead experiments to this directory instead of deleting them')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Map commands to functions
    commands = {
        'list': list_experiments,
        'export': export_experiments,
        'show': show_experiment,
        'compare': compare_exps,
        'find': find_exp_by_config,
        'hash': hash_config_file,
        'create-metrics': create_metrics_files,
        'cleanup': cleanup_dead_experiments
    }
    
    # Execute the command
    commands[args.command](args)

if __name__ == "__main__":
    main() 