#!/usr/bin/env python3
"""
manage_feature_sets.py

CLI tool for managing multiple feature sets. Allows you to:
- List all feature sets
- Create new feature sets
- Copy feature sets
- View feature set information
- Delete feature sets (with confirmation)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from feature_set_manager import (
    list_feature_sets,
    feature_set_exists,
    create_feature_set,
    get_feature_set_info,
    validate_feature_set,
    get_all_feature_sets,
    get_feature_set_config_path,
    get_feature_set_data_path,
    get_train_features_config_path,
    DEFAULT_FEATURE_SET
)


def cmd_list(args):
    """List all available feature sets with enhanced details."""
    all_sets = get_all_feature_sets()
    
    print("\n" + "="*80)
    print("AVAILABLE FEATURE SETS")
    print("="*80)
    
    if not all_sets:
        print("No feature sets found.")
        return
    
    for info in all_sets:
        if 'error' in info:
            print(f"\n{info['name']}: ERROR - {info['error']}")
            continue
        
        fs = info['name']
        is_default = " (default)" if fs == DEFAULT_FEATURE_SET else ""
        validation = info.get('validation', {})
        is_valid = validation.get('is_valid', False)
        status = "[OK]" if is_valid else "[WARN]"
        
        print(f"\n{status} {fs}{is_default}:")
        print(f"  Config: {info['config_path']} {'[OK]' if info['config_exists'] else '[MISSING]'}")
        print(f"  Data: {info['data_path']} {'[OK]' if info['data_exists'] else '[MISSING]'}")
        print(f"  Train Config: {'[OK]' if info['train_config_exists'] else '[MISSING]'}")
        
        if 'enabled_features' in info:
            print(f"  Features: {info['enabled_features']} enabled / {info['total_features']} total")
        
        if 'data_files' in info:
            print(f"  Data files: {info['data_files']} tickers")
        
        if 'metadata' in info and 'description' in info['metadata']:
            print(f"  Description: {info['metadata']['description']}")
        
        # Show validation warnings/errors
        if validation.get('warnings'):
            for warning in validation['warnings']:
                print(f"  [WARN] {warning}")
        if validation.get('errors'):
            for error in validation['errors']:
                print(f"  [ERROR] {error}")
    
    print("\n" + "="*80)


def cmd_create(args):
    """Create a new feature set."""
    if feature_set_exists(args.name):
        print(f"Error: Feature set '{args.name}' already exists.")
        print(f"Use 'list' to see all feature sets.")
        sys.exit(1)
    
    try:
        create_feature_set(
            feature_set=args.name,
            copy_from=args.copy_from,
            description=args.description
        )
        print(f"\n[OK] Feature set '{args.name}' created successfully!")
        
        # Show info
        info = get_feature_set_info(args.name)
        print(f"\nFeature set info:")
        print(f"  Config: {info['config_path']}")
        print(f"  Data: {info['data_path']}")
        if args.copy_from:
            print(f"  Copied from: {args.copy_from}")
    except Exception as e:
        print(f"Error creating feature set: {e}")
        sys.exit(1)


def cmd_info(args):
    """Show detailed information about a feature set."""
    if not feature_set_exists(args.name):
        print(f"Error: Feature set '{args.name}' does not exist.")
        print(f"Use 'list' to see all feature sets.")
        sys.exit(1)
    
    try:
        info = get_feature_set_info(args.name)
        validation = validate_feature_set(args.name)
        
        print("\n" + "="*80)
        print(f"FEATURE SET: {args.name}")
        if args.name == DEFAULT_FEATURE_SET:
            print("(DEFAULT)")
        print("="*80)
        print(f"Config path: {info['config_path']}")
        print(f"Data path: {info['data_path']}")
        print(f"Train config path: {info['train_config_path']}")
        print(f"\nStatus:")
        print(f"  Config exists: {info['config_exists']} {'[OK]' if info['config_exists'] else '[MISSING]'}")
        print(f"  Data exists: {info['data_exists']} {'[OK]' if info['data_exists'] else '[MISSING]'}")
        print(f"  Train config exists: {info['train_config_exists']} {'[OK]' if info['train_config_exists'] else '[MISSING]'}")
        print(f"  Valid: {validation['is_valid']} {'[OK]' if validation['is_valid'] else '[INVALID]'}")
        
        if 'enabled_features' in info:
            print(f"\nFeatures:")
            print(f"  Enabled: {info['enabled_features']}")
            print(f"  Total: {info['total_features']}")
        
        if 'data_files' in info:
            print(f"\nData:")
            print(f"  Tickers: {info['data_files']}")
        
        if 'metadata' in info:
            print(f"\nMetadata:")
            for key, value in info['metadata'].items():
                print(f"  {key}: {value}")
        
        if validation.get('warnings'):
            print(f"\nWarnings:")
            for warning in validation['warnings']:
                print(f"  [WARN] {warning}")
        
        if validation.get('errors'):
            print(f"\nErrors:")
            for error in validation['errors']:
                print(f"  [ERROR] {error}")
        
        print("="*80 + "\n")
    except Exception as e:
        print(f"Error getting feature set info: {e}")
        sys.exit(1)


def cmd_delete(args):
    """Delete a feature set with optional cleanup options."""
    if args.name == DEFAULT_FEATURE_SET:
        print(f"Error: Cannot delete default feature set '{DEFAULT_FEATURE_SET}'")
        sys.exit(1)
    
    if not feature_set_exists(args.name):
        print(f"Error: Feature set '{args.name}' does not exist.")
        print(f"Use 'list' to see all feature sets.")
        sys.exit(1)
    
    # Show what will be deleted
    config_path = get_feature_set_config_path(args.name)
    data_path = get_feature_set_data_path(args.name)
    train_config_path = get_train_features_config_path(args.name)
    
    print(f"\n[WARNING] This will delete feature set '{args.name}':")
    print(f"  Config: {config_path}")
    print(f"  Train Config: {train_config_path}")
    if args.delete_data:
        print(f"  Data directory: {data_path}")
    else:
        print(f"  Data directory: {data_path} (will be kept)")
    
    # Confirmation
    if not args.force:
        response = input("\nAre you sure you want to delete this feature set? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    # Delete files
    import shutil
    deleted = []
    
    try:
        if config_path.exists():
            config_path.unlink()
            deleted.append(f"Config: {config_path}")
        
        if train_config_path.exists():
            train_config_path.unlink()
            deleted.append(f"Train config: {train_config_path}")
        
        if args.delete_data and data_path.exists():
            shutil.rmtree(data_path)
            deleted.append(f"Data directory: {data_path}")
        
        # Try to delete feature implementation directory
        feature_dir = Path(__file__).parent.parent / "features" / "sets" / args.name
        if feature_dir.exists():
            shutil.rmtree(feature_dir)
            deleted.append(f"Feature implementation: {feature_dir}")
        
        print(f"\n[OK] Feature set '{args.name}' deleted successfully!")
        print("Deleted:")
        for item in deleted:
            print(f"  - {item}")
        
    except Exception as e:
        print(f"Error deleting feature set: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage multiple feature sets for experimentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all feature sets
  python src/manage_feature_sets.py list

  # Create a new feature set (copies from default)
  python src/manage_feature_sets.py create v2 --description "Experimental feature set"

  # Create a new feature set by copying from another
  python src/manage_feature_sets.py create v3 --copy-from v2 --description "Refined version"

  # View feature set information
  python src/manage_feature_sets.py info v2

  # Delete a feature set (keeps data)
  python src/manage_feature_sets.py delete v2

  # Delete a feature set including data
  python src/manage_feature_sets.py delete v2 --delete-data
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all feature sets")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new feature set")
    create_parser.add_argument("name", help="Name of the new feature set (e.g., 'v2', 'experimental')")
    create_parser.add_argument("--copy-from", default=None, help=f"Feature set to copy from (default: '{DEFAULT_FEATURE_SET}')")
    create_parser.add_argument("--description", default=None, help="Description of the feature set")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed information about a feature set")
    info_parser.add_argument("name", help="Name of the feature set")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a feature set")
    delete_parser.add_argument("name", help="Name of the feature set to delete")
    delete_parser.add_argument("--delete-data", action="store_true", help="Also delete the data directory")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "list":
        cmd_list(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "delete":
        cmd_delete(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

