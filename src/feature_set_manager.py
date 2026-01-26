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
import sys
from pathlib import Path
from typing import Optional, Dict, List
import yaml

# —— CONFIGURATION —— #
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FEATURE_SET = "v1"  # Default feature set name

# Ensure project root is in Python path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_feature_set_config_path(feature_set: str) -> Path:
    """
    Get the config file path for a feature set.
    
    Args:
        feature_set: Name of the feature set (e.g., "v1", "v2", "baseline")
    
    Returns:
        Path to the feature set config file
    """
    # All feature sets now use the features_<set>.yaml naming convention
    return CONFIG_DIR / f"features_{feature_set}.yaml"


def get_feature_set_data_path(feature_set: str) -> Path:
    """
    Get the data directory path for a feature set.
    
    Args:
        feature_set: Name of the feature set (e.g., "v1", "v2", "baseline")
    
    Returns:
        Path to the feature set data directory
    """
    # All feature sets now use the features_labeled_<set> naming convention
    return DATA_DIR / f"features_labeled_{feature_set}"


def get_train_features_config_path(feature_set: str) -> Path:
    """
    Get the train features config file path for a feature set.
    
    Args:
        feature_set: Name of the feature set
    
    Returns:
        Path to the train features config file
    """
    # All feature sets now use the train_features_<set>.yaml naming convention
    return CONFIG_DIR / f"train_features_{feature_set}.yaml"


def list_feature_sets() -> List[str]:
    """
    List all available feature sets.
    
    Returns:
        List of feature set names
    """
    feature_sets = []
    
    # Find all feature config files
    for config_file in CONFIG_DIR.glob("features_*.yaml"):
        # Extract feature set name from filename (e.g., "features_v1.yaml" -> "v1")
        set_name = config_file.stem.replace("features_", "")
        if set_name and set_name not in feature_sets:
            # Verify the registry module exists
            try:
                import importlib
                # Convert feature set name to valid Python module name
                # Replace spaces and special chars with underscores
                module_name = set_name.replace(" ", "_").replace("-", "_")
                # Try importing with the sanitized name
                importlib.import_module(f"features.sets.{module_name}.registry")
                feature_sets.append(set_name)
            except (ImportError, SyntaxError):
                # Skip if registry doesn't exist or has syntax errors
                pass
    
    return sorted(feature_sets) if feature_sets else [DEFAULT_FEATURE_SET]


def feature_set_exists(feature_set: str) -> bool:
    """
    Check if a feature set exists.
    
    Args:
        feature_set: Name of the feature set
    
    Returns:
        True if the feature set exists, False otherwise
    """
    # Check if config file exists
    config_path = get_feature_set_config_path(feature_set)
    if not config_path.exists():
        return False
    
    # Also check if the registry module exists
    try:
        import importlib
        # Ensure project root is in path for imports
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        # Convert feature set name to valid Python module name
        module_name = feature_set.replace(" ", "_").replace("-", "_")
        importlib.import_module(f"features.sets.{module_name}.registry")
        return True
    except ImportError as e:
        # Debug: print error if needed (can be removed later)
        # print(f"Debug: Failed to import registry for {feature_set}: {e}")
        return False


def create_feature_set(
    feature_set: str,
    copy_from: Optional[str] = None,
    description: Optional[str] = None,
    from_scratch: bool = False
) -> bool:
    """
    Create a new feature set.
    
    Args:
        feature_set: Name of the new feature set
        copy_from: Optional name of feature set to copy from (default: DEFAULT_FEATURE_SET if not from_scratch)
        description: Optional description to add to config file
        from_scratch: If True, create empty feature set with no features enabled
    
    Returns:
        True if created successfully, False otherwise
    """
    if feature_set == DEFAULT_FEATURE_SET:
        raise ValueError(f"Cannot create feature set '{DEFAULT_FEATURE_SET}' (it's the default)")
    
    if feature_set_exists(feature_set):
        raise ValueError(f"Feature set '{feature_set}' already exists")
    
    # If creating from scratch, skip copying
    if from_scratch:
        return _create_feature_set_from_scratch(feature_set, description)
    
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
    
    # Copy feature implementation directory (technical.py and registry.py)
    # Use sanitized names for directories to match Python module naming
    source_sanitized = source_set.replace(" ", "_").replace("-", "_")
    target_sanitized = feature_set.replace(" ", "_").replace("-", "_")
    source_feature_dir = PROJECT_ROOT / "features" / "sets" / source_sanitized
    target_feature_dir = PROJECT_ROOT / "features" / "sets" / target_sanitized
    
    if source_feature_dir.exists() and source_feature_dir.is_dir():
        # Create target directory
        target_feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy __init__.py if it exists
        source_init = source_feature_dir / "__init__.py"
        if source_init.exists():
            shutil.copy2(source_init, target_feature_dir / "__init__.py")
        
        # Copy technical.py
        source_technical = source_feature_dir / "technical.py"
        if source_technical.exists():
            shutil.copy2(source_technical, target_feature_dir / "technical.py")
        
        # Copy registry.py
        source_registry = source_feature_dir / "registry.py"
        if source_registry.exists():
            # Read and update imports in registry.py
            with open(source_registry, 'r', encoding='utf-8') as f:
                registry_content = f.read()
            
            # Update import statement from old set to new set (use sanitized names)
            registry_content = registry_content.replace(
                f"from features.sets.{source_sanitized}.technical import",
                f"from features.sets.{target_sanitized}.technical import"
            )
            
            # Write updated registry.py
            with open(target_feature_dir / "registry.py", 'w', encoding='utf-8') as f:
                f.write(registry_content)
    
    return True


def _create_feature_set_from_scratch(feature_set: str, description: Optional[str] = None) -> bool:
    """
    Create a new feature set from scratch with empty structure.
    
    Args:
        feature_set: Name of the new feature set
        description: Optional description to add to config file
    
    Returns:
        True if created successfully, False otherwise
    """
    # Create empty config file
    target_config = get_feature_set_config_path(feature_set)
    target_config.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        'features': {}
    }
    
    # Add metadata
    if description:
        config['metadata'] = {
            'description': description,
            'feature_set': feature_set,
            'created_from': 'scratch'
        }
    else:
        config['metadata'] = {
            'feature_set': feature_set,
            'created_from': 'scratch'
        }
    
    with open(target_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Create empty train_features config
    target_train_config = get_train_features_config_path(feature_set)
    train_config = {
        'features': {}
    }
    with open(target_train_config, 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Create feature implementation directory with minimal files
    # Use sanitized name for directory (spaces/dashes to underscores) to match Python module naming
    sanitized_name = feature_set.replace(" ", "_").replace("-", "_")
    target_feature_dir = PROJECT_ROOT / "features" / "sets" / sanitized_name
    target_feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_content = f"""# Feature set: {feature_set}
"""
    with open(target_feature_dir / "__init__.py", 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    # Create minimal technical.py
    technical_content = f"""# features/sets/{feature_set}/technical.py

\"\"\"
Technical feature functions for feature set '{feature_set}'.

This file is empty by default. Add your feature functions here.
Each function should take a DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
and return a pandas Series.
\"\"\"

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from typing import Optional

# Import shared utilities
from features.shared.utils import (
    _load_spy_data,
    _get_column,
    _get_close_series,
    _get_open_series,
    _get_high_series,
    _get_low_series,
    _get_volume_series,
    _rolling_percentile_rank,
)

# Add your feature functions here
# Example:
# def feature_my_custom_feature(df: DataFrame) -> Series:
#     \"\"\"My custom feature.\"\"\"
#     close = _get_close_series(df)
#     return close * 2
"""
    with open(target_feature_dir / "technical.py", 'w', encoding='utf-8') as f:
        f.write(technical_content)
    
    # Create minimal registry.py
    registry_content = f"""# features/sets/{sanitized_name}/registry.py

\"\"\"
Feature registry for feature set '{feature_set}'.

Maps feature names (as used in YAML configs) to feature-extraction functions
defined in technical.py.
\"\"\"

# Import feature functions from technical.py
# Uncomment and add your imports here as you create feature functions:
# from features.sets.{sanitized_name}.technical import (
#     feature_my_custom_feature,
# )

# Feature registry mapping
# Add your features here as you create them:
FEATURE_REGISTRY = dict()
# Example:
# FEATURE_REGISTRY = {{
#     'my_custom_feature': feature_my_custom_feature,
# }}


def load_enabled_features(config_path: str):
    \"\"\"
    Load enabled features from YAML config file.
    
    Args:
        config_path: Path to features.yaml config file
    
    Returns:
        Dictionary mapping feature names to functions
    \"\"\"
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {{}}
    
    features = config.get('features', {{}})
    enabled = {{}}
    
    for name, enabled_flag in features.items():
        if enabled_flag == 1 and name in FEATURE_REGISTRY:
            enabled[name] = FEATURE_REGISTRY[name]
    
    return enabled
"""
    with open(target_feature_dir / "registry.py", 'w', encoding='utf-8') as f:
        f.write(registry_content)
    
    return True


def update_feature_set_metadata(
    feature_set: str,
    description: Optional[str] = None
) -> bool:
    """
    Update metadata for an existing feature set.
    
    Args:
        feature_set: Name of the feature set to update
        description: New description (None to remove description)
    
    Returns:
        True if updated successfully, False otherwise
    """
    if not feature_set_exists(feature_set):
        raise ValueError(f"Feature set '{feature_set}' does not exist")
    
    config_path = get_feature_set_config_path(feature_set)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Ensure metadata section exists
    if 'metadata' not in config:
        config['metadata'] = {}
    
    # Update description
    if description is not None:
        config['metadata']['description'] = description
    elif 'description' in config['metadata']:
        # Remove description if explicitly set to None
        del config['metadata']['description']
    
    # Ensure feature_set name is in metadata
    config['metadata']['feature_set'] = feature_set
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return True


def rename_feature_set(old_name: str, new_name: str) -> bool:
    """
    Rename a feature set (moves all files and directories).
    
    Args:
        old_name: Current name of the feature set
        new_name: New name for the feature set
    
    Returns:
        True if renamed successfully, False otherwise
    """
    if old_name == DEFAULT_FEATURE_SET:
        raise ValueError(f"Cannot rename default feature set '{DEFAULT_FEATURE_SET}'")
    
    if not feature_set_exists(old_name):
        raise ValueError(f"Feature set '{old_name}' does not exist")
    
    if feature_set_exists(new_name):
        raise ValueError(f"Feature set '{new_name}' already exists")
    
    # Get paths
    old_config = get_feature_set_config_path(old_name)
    new_config = get_feature_set_config_path(new_name)
    
    old_train_config = get_train_features_config_path(old_name)
    new_train_config = get_train_features_config_path(new_name)
    
    old_data_dir = get_feature_set_data_path(old_name)
    new_data_dir = get_feature_set_data_path(new_name)
    
    old_feature_dir = PROJECT_ROOT / "features" / "sets" / old_name
    new_feature_dir = PROJECT_ROOT / "features" / "sets" / new_name
    
    # Move config file
    if old_config.exists():
        new_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_config), str(new_config))
        
        # Update metadata in config
        with open(new_config, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        if 'metadata' not in config:
            config['metadata'] = {}
        config['metadata']['feature_set'] = new_name
        
        with open(new_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Move train config file
    if old_train_config.exists():
        new_train_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_train_config), str(new_train_config))
    
    # Move data directory
    if old_data_dir.exists():
        new_data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_data_dir), str(new_data_dir))
    
    # Move feature implementation directory
    if old_feature_dir.exists():
        new_feature_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_feature_dir), str(new_feature_dir))
        
        # Update imports in registry.py
        registry_file = new_feature_dir / "registry.py"
        if registry_file.exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry_content = f.read()
            
            # Update import statement
            registry_content = registry_content.replace(
                f"from features.sets.{old_name}.technical import",
                f"from features.sets.{new_name}.technical import"
            )
            
            with open(registry_file, 'w', encoding='utf-8') as f:
                f.write(registry_content)
    
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


def validate_feature_set(feature_set: str) -> Dict[str, any]:
    """
    Validate a feature set and return validation results.
    
    Args:
        feature_set: Name of the feature set to validate
    
    Returns:
        Dictionary with validation results including:
        - is_valid: bool
        - errors: List of error messages
        - warnings: List of warning messages
        - missing_files: List of missing file paths
    """
    errors = []
    warnings = []
    missing_files = []
    
    # Check if feature set exists
    if not feature_set_exists(feature_set):
        errors.append(f"Feature set '{feature_set}' does not exist")
        return {
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'missing_files': missing_files
        }
    
    # Check config file
    config_path = get_feature_set_config_path(feature_set)
    if not config_path.exists():
        errors.append(f"Config file missing: {config_path}")
        missing_files.append(str(config_path))
    else:
        # Validate config file can be loaded
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                if 'features' not in config:
                    warnings.append("Config file has no 'features' section")
        except Exception as e:
            errors.append(f"Config file cannot be loaded: {e}")
    
    # Check train config file
    train_config_path = get_train_features_config_path(feature_set)
    if not train_config_path.exists():
        warnings.append(f"Train config file missing: {train_config_path}")
        missing_files.append(str(train_config_path))
    
    # Check data directory
    data_path = get_feature_set_data_path(feature_set)
    if not data_path.exists():
        warnings.append(f"Data directory missing: {data_path}")
    else:
        parquet_files = list(data_path.glob("*.parquet"))
        if len(parquet_files) == 0:
            warnings.append(f"Data directory exists but has no parquet files")
    
    # Check registry module
    try:
        import importlib
        # Convert feature set name to valid Python module name
        module_name = feature_set.replace(" ", "_").replace("-", "_")
        registry_module = importlib.import_module(f"features.sets.{module_name}.registry")
        if not hasattr(registry_module, 'load_enabled_features'):
            errors.append(f"Registry module missing 'load_enabled_features' function")
    except (ImportError, SyntaxError) as e:
        errors.append(f"Registry module cannot be imported: {e}")
    
    # Check technical module
    try:
        import importlib
        # Convert feature set name to valid Python module name
        module_name = feature_set.replace(" ", "_").replace("-", "_")
        importlib.import_module(f"features.sets.{module_name}.technical")
    except (ImportError, SyntaxError) as e:
        errors.append(f"Technical module cannot be imported: {e}")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'missing_files': missing_files
    }


def get_all_feature_sets() -> List[Dict]:
    """
    Get all feature sets with their information.
    
    Returns:
        List of dictionaries, each containing feature set information
    """
    feature_set_names = list_feature_sets()
    result = []
    
    for name in feature_set_names:
        try:
            info = get_feature_set_info(name)
            validation = validate_feature_set(name)
            info['validation'] = validation
            result.append(info)
        except Exception as e:
            result.append({
                'name': name,
                'error': str(e)
            })
    
    return result

