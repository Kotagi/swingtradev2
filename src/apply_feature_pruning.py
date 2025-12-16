#!/usr/bin/env python3
"""
apply_feature_pruning.py

Applies feature pruning recommendations by updating config/train_features.yaml
to disable low-value and redundant features.

This script:
1. Loads pruning recommendations from JSON
2. Updates config/train_features.yaml to disable recommended features
3. Creates a backup of the original config
4. Shows what will be changed before applying
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
TRAIN_FEATURES_CONFIG = CONFIG_DIR / "train_features.yaml"
RECOMMENDATIONS_FILE = PROJECT_ROOT / "models" / "feature_pruning_recommendations.json"


def load_recommendations(recommendations_file: Path) -> dict:
    """Load pruning recommendations from JSON file."""
    if not recommendations_file.exists():
        raise FileNotFoundError(f"Recommendations file not found: {recommendations_file}")
    
    with open(recommendations_file, 'r') as f:
        return json.load(f)


def load_config(config_file: Path) -> dict:
    """Load YAML config file."""
    import yaml
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict, config_file: Path):
    """Save YAML config file."""
    import yaml
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def apply_pruning(
    config_file: Path,
    recommendations_file: Path,
    dry_run: bool = False,
    create_backup: bool = True
) -> dict:
    """
    Apply pruning recommendations to config file.
    
    Args:
        config_file: Path to train_features.yaml
        recommendations_file: Path to pruning recommendations JSON
        dry_run: If True, don't actually modify files (just show what would change)
        create_backup: If True, create backup of original config
    
    Returns:
        Dictionary with summary of changes
    """
    # Load recommendations
    recommendations = load_recommendations(recommendations_file)
    features_to_remove = set(recommendations.get("features_to_remove", []))
    
    if not features_to_remove:
        print("No features recommended for removal.")
        return {"removed": 0, "kept": 0, "not_found": []}
    
    # Load current config
    config = load_config(config_file)
    features = config.get("features", {})
    
    # Track changes
    removed = []
    not_found = []
    already_disabled = []
    
    # Apply pruning
    for feat in features_to_remove:
        if feat in features:
            if features[feat] == 1:
                removed.append(feat)
                if not dry_run:
                    features[feat] = 0  # Disable feature
            else:
                already_disabled.append(feat)
        else:
            not_found.append(feat)
    
    # Create backup if not dry run
    if not dry_run and create_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = config_file.with_suffix(f".yaml.backup_{timestamp}")
        shutil.copy2(config_file, backup_file)
        print(f"Backup created: {backup_file}")
    
    # Save updated config
    if not dry_run:
        config["features"] = features
        save_config(config, config_file)
        print(f"Config updated: {config_file}")
    
    return {
        "removed": len(removed),
        "kept": len(features) - len(removed),
        "not_found": not_found,
        "already_disabled": already_disabled,
        "removed_features": sorted(removed)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply feature pruning recommendations to config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(TRAIN_FEATURES_CONFIG),
        help="Path to train_features.yaml config file"
    )
    parser.add_argument(
        "--recommendations",
        type=str,
        default=str(RECOMMENDATIONS_FILE),
        help="Path to pruning recommendations JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually modifying files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original config file"
    )
    
    args = parser.parse_args()
    
    config_file = Path(args.config)
    recommendations_file = Path(args.recommendations)
    
    print("="*80)
    print("APPLY FEATURE PRUNING")
    print("="*80)
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No files will be modified\n")
    
    # Load recommendations to show summary
    recommendations = load_recommendations(recommendations_file)
    features_to_remove = recommendations.get("features_to_remove", [])
    
    print(f"\nRecommendations loaded: {len(features_to_remove)} features to remove")
    print(f"Config file: {config_file}")
    print(f"Recommendations file: {recommendations_file}")
    
    # Apply pruning
    result = apply_pruning(
        config_file,
        recommendations_file,
        dry_run=args.dry_run,
        create_backup=not args.no_backup
    )
    
    # Print summary
    print("\n" + "="*80)
    print("PRUNING SUMMARY")
    print("="*80)
    print(f"Features disabled: {result['removed']}")
    print(f"Features kept: {result['kept']}")
    
    if result['not_found']:
        print(f"\nFeatures not found in config ({len(result['not_found'])}):")
        for feat in result['not_found'][:10]:
            print(f"  - {feat}")
        if len(result['not_found']) > 10:
            print(f"  ... and {len(result['not_found']) - 10} more")
    
    if result['already_disabled']:
        print(f"\nFeatures already disabled ({len(result['already_disabled'])}):")
        for feat in result['already_disabled'][:10]:
            print(f"  - {feat}")
        if len(result['already_disabled']) > 10:
            print(f"  ... and {len(result['already_disabled']) - 10} more")
    
    if result['removed'] > 0:
        print(f"\nFeatures disabled ({len(result['removed_features'])}):")
        for feat in result['removed_features']:
            print(f"  - {feat}")
    
    if args.dry_run:
        print("\n[DRY RUN] - Run without --dry-run to apply changes")
    else:
        print(f"\n✅ Pruning applied! Config updated: {config_file}")
        print(f"   New feature count: {result['kept']} (reduced from {result['removed'] + result['kept']})")
        print("\nNext steps:")
        print("  1. Review the updated config/train_features.yaml")
        print("  2. Retrain the model: python src/swing_trade_app.py train --tune")
        print("  3. Compare performance with backtest")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

