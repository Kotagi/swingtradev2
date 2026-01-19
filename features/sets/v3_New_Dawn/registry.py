# features/sets/v3_New_Dawn/registry.py

"""
Feature registry for feature set 'v3_New_Dawn'.

Maps feature names (as used in YAML configs) to feature-extraction functions
defined in technical.py.
"""

# Import feature functions from technical.py
# Uncomment and add your imports here as you create feature functions:
# from features.sets.v3_New_Dawn.technical import (
#     feature_my_custom_feature,
# )

# Feature registry mapping
# Add your features here as you create them:
FEATURE_REGISTRY = dict()
# Example:
# FEATURE_REGISTRY = {
#     'my_custom_feature': feature_my_custom_feature,
# }


def load_enabled_features(config_path: str):
    """
    Load enabled features from YAML config file.
    
    Args:
        config_path: Path to features.yaml config file
    
    Returns:
        Dictionary mapping feature names to functions
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    features = config.get('features', {})
    enabled = {}
    
    for name, enabled_flag in features.items():
        if enabled_flag == 1 and name in FEATURE_REGISTRY:
            enabled[name] = FEATURE_REGISTRY[name]
    
    return enabled
