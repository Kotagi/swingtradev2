#!/usr/bin/env python3
"""
feature_set_manager.py

Utility module for managing multiple feature sets. Allows you to:
- Create new feature sets
- List existing feature sets
- Copy feature sets
- Get paths for feature set configs and data directories
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, List
import yaml

# —— CONFIGURATION —— #
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FEATURE_SET = "v1"  # Default feature set name


def get_feature_set_config_path(feature_set: str) -> Path:
    """
    Get the config file path for a feature set.
    
    Args:
        feature_set: Name of the feature set (e.g., "v1", "v2", "baseline")
    
    Returns:
        Path to the feature set config file
    """
    if feature_set == DEFAULT_FEATURE_SET:
        # Default set uses the original features.yaml for backward compatibility
        return CONFIG_DIR / "features.yaml"
    else:
        return CONFIG_DIR / f"features_{feature_set}.yaml"


def get_feature_set_data_path(feature_set: str) -> Path:
    """
    Get the data directory path for a feature set.
    
    Args:
        feature_set: Name of the feature set (e.g., "v1", "v2", "baseline")
    
    Returns:
        Path to the feature set data directory
    """
    if feature_set == DEFAULT_FEATURE_SET:
        # Default set uses the original directory for backward compatibility
        return DATA_DIR / "features_labeled"
    else:
        return DATA_DIR / f"features_labeled_{feature_set}"


def get_train_features_config_path(feature_set: str) -> Path:
    """
    Get the train features config file path for a feature set.
    
    Args:
        feature_set: Name of the feature set
    
    Returns:
        Path to the train features config file
    """
    if feature_set == DEFAULT_FEATURE_SET:
        # Default set uses the original train_features.yaml for backward compatibility
        return CONFIG_DIR / "train_features.yaml"
    else:
        return CONFIG_DIR / f"train_features_{feature_set}.yaml"


def list_feature_sets() -> List[str]:
    """
    List all available feature sets.
    
    Returns:
        List of feature set names
    """
    feature_sets = [DEFAULT_FEATURE_SET]  # Always include default
    
    # Find all feature config files
    for config_file in CONFIG_DIR.glob("features_*.yaml"):
        # Extract feature set name from filename (e.g., "features_v2.yaml" -> "v2")
        set_name = config_file.stem.replace("features_", "")
        if set_name and set_name not in feature_sets:
            feature_sets.append(set_name)
    
    return sorted(feature_sets)


def feature_set_exists(feature_set: str) -> bool:
    """
    Check if a feature set exists.
    
    Args:
        feature_set: Name of the feature set
    
    Returns:
        True if the feature set exists, False otherwise
    """
    if feature_set == DEFAULT_FEATURE_SET:
        # Default set exists if features.yaml exists
        return (CONFIG_DIR / "features.yaml").exists()
    else:
        return get_feature_set_config_path(feature_set).exists()


def create_feature_set(
    feature_set: str,
    copy_from: Optional[str] = None,
    description: Optional[str] = None
) -> bool:
    """
    Create a new feature set.
    
    Args:
        feature_set: Name of the new feature set
        copy_from: Optional name of feature set to copy from (default: DEFAULT_FEATURE_SET)
        description: Optional description to add to config file
    
    Returns:
        True if created successfully, False otherwise
    """
    if feature_set == DEFAULT_FEATURE_SET:
        raise ValueError(f"Cannot create feature set '{DEFAULT_FEATURE_SET}' (it's the default)")
    
    if feature_set_exists(feature_set):
        raise ValueError(f"Feature set '{feature_set}' already exists")
    
    # Determine source feature set
    source_set = copy_from if copy_from else DEFAULT_FEATURE_SET
    
    if not feature_set_exists(source_set):
        raise ValueError(f"Source feature set '{source_set}' does not exist")
    
    # Copy config file
    source_config = get_feature_set_config_path(source_set)
    target_config = get_feature_set_config_path(feature_set)
    
    if not source_config.exists():
        raise FileNotFoundError(f"Source config file not found: {source_config}")
    
    # Load source config
    with open(source_config, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Add description if provided
    if description:
        if 'metadata' not in config:
            config['metadata'] = {}
        config['metadata']['description'] = description
        config['metadata']['feature_set'] = feature_set
        config['metadata']['created_from'] = source_set
    
    # Save new config
    target_config.parent.mkdir(parents=True, exist_ok=True)
    with open(target_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Copy train_features config if it exists
    source_train_config = get_train_features_config_path(source_set)
    target_train_config = get_train_features_config_path(feature_set)
    
    if source_train_config.exists():
        shutil.copy2(source_train_config, target_train_config)
    
    return True


def get_feature_set_info(feature_set: str) -> Dict:
    """
    Get information about a feature set.
    
    Args:
        feature_set: Name of the feature set
    
    Returns:
        Dictionary with feature set information
    """
    if not feature_set_exists(feature_set):
        raise ValueError(f"Feature set '{feature_set}' does not exist")
    
    config_path = get_feature_set_config_path(feature_set)
    data_path = get_feature_set_data_path(feature_set)
    train_config_path = get_train_features_config_path(feature_set)
    
    info = {
        'name': feature_set,
        'config_path': str(config_path),
        'data_path': str(data_path),
        'train_config_path': str(train_config_path),
        'config_exists': config_path.exists(),
        'data_exists': data_path.exists(),
        'train_config_exists': train_config_path.exists(),
    }
    
    # Load metadata if available
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                if 'metadata' in config:
                    info['metadata'] = config['metadata']
        except Exception:
            pass
    
    # Count features in config
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                features = config.get('features', {})
                enabled_features = sum(1 for v in features.values() if v == 1)
                total_features = len(features)
                info['enabled_features'] = enabled_features
                info['total_features'] = total_features
        except Exception:
            pass
    
    # Count data files
    if data_path.exists():
        parquet_files = list(data_path.glob("*.parquet"))
        info['data_files'] = len(parquet_files)
    else:
        info['data_files'] = 0
    
    return info

