#!/usr/bin/env python3
"""
prune_features_config.py

Prune low/zero-importance features by disabling them in the build and training configs.
Updates both features_<set>.yaml (build) and train_features_<set>.yaml (training).
"""

import sys
import joblib
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_script_file = Path(__file__).resolve()
PROJECT_ROOT = _script_file.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_DIR = PROJECT_ROOT / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_feature_set_config_path(feature_set: str) -> Path:
    """Path to features_<set>.yaml (build config)."""
    return CONFIG_DIR / f"features_{feature_set}.yaml"


def get_train_features_config_path(feature_set: str) -> Path:
    """Path to train_features_<set>.yaml (training config)."""
    return CONFIG_DIR / f"train_features_{feature_set}.yaml"


def get_enabled_feature_names(feature_set: str, config_type: str = "train") -> List[str]:
    """
    Return list of feature names that are enabled (1) in the given config.

    Args:
        feature_set: e.g. "v3_New_Dawn"
        config_type: "train" (train_features_<set>.yaml) or "build" (features_<set>.yaml)

    Returns:
        List of feature names with value 1.
    """
    path = get_train_features_config_path(feature_set) if config_type == "train" else get_feature_set_config_path(feature_set)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "features" not in data:
        return []
    return [k for k, v in data["features"].items() if v == 1]


def get_importances_from_model(model_path: str) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Load feature importances from a model pickle or from training_metadata.json.

    Args:
        model_path: Path to .pkl or to models dir (will use training_metadata.json).

    Returns:
        (importances_dict: name -> importance, feature_set from metadata or None)
    """
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    feature_set = None
    importances = {}

    # Try pickle first
    if path.suffix.lower() == ".pkl" and path.exists():
        data = joblib.load(path)
        if isinstance(data, dict):
            meta = data.get("metadata", {})
            feature_set = meta.get("feature_set")
            importances = meta.get("feature_importances", {})
        if not importances:
            return {}, feature_set
        return {k: float(v) for k, v in importances.items()}, feature_set

    # Try training_metadata.json in same dir or models/
    if path.is_dir():
        meta_file = path / "training_metadata.json"
    else:
        meta_file = path.parent / "training_metadata.json"
    if meta_file.exists():
        import json
        with open(meta_file, "r") as f:
            meta = json.load(f)
        feature_set = meta.get("feature_set")
        importances = meta.get("feature_importances", {})
        if importances:
            return {k: float(v) for k, v in importances.items()}, feature_set

    return {}, feature_set


def compute_to_disable(
    importances: Dict[str, float],
    rule: str,
    rule_param: float,
    config_feature_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Compute which features to disable (set to 0) based on importance rule.

    Args:
        importances: feature name -> importance (>= 0).
        rule: "drop_zero" | "keep_top_n" | "drop_below_threshold"
        rule_param: For keep_top_n: N (int). For drop_below_threshold: min importance.
        config_feature_names: If set, only consider features in this list (e.g. currently enabled).

    Returns:
        List of feature names to disable.
    """
    if not importances:
        return []

    # Restrict to config names if provided
    if config_feature_names:
        imp = {k: v for k, v in importances.items() if k in config_feature_names}
    else:
        imp = dict(importances)

    if not imp:
        return []

    # Sort by importance descending (highest first)
    sorted_names = sorted(imp.keys(), key=lambda x: imp[x], reverse=True)

    if rule == "drop_zero":
        return [n for n in sorted_names if imp[n] <= 0 or (imp[n] < 1e-10)]
    if rule == "keep_top_n":
        n = max(0, int(rule_param))
        if n >= len(sorted_names):
            return []
        return sorted_names[n:]
    if rule == "drop_below_threshold":
        thresh = float(rule_param)
        return [n for n in sorted_names if imp[n] < thresh]

    return []


def apply_prune(
    feature_set: str,
    to_disable: List[str],
    dry_run: bool = False,
    update_build_config: bool = True,
    update_train_config: bool = True,
) -> Tuple[int, str]:
    """
    Set given features to 0 in build and/or train config YAMLs.

    Only updates features that exist in the config and are currently 1 (enabled).

    Args:
        feature_set: e.g. "v3_New_Dawn"
        to_disable: List of feature names to set to 0.
        dry_run: If True, do not write files; return count that would be changed.
        update_build_config: Update features_<set>.yaml.
        update_train_config: Update train_features_<set>.yaml.

    Returns:
        (number of features actually set to 0, message)
    """
    if not to_disable:
        return 0, "No features to disable."

    to_disable_set = set(to_disable)
    total_changed = 0
    messages = []

    def update_one(cfg_path: Path) -> int:
        if not cfg_path.exists():
            return 0
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "features" not in data:
            return 0
        features = data["features"]
        changed = 0
        for name in to_disable_set:
            if name in features and features[name] == 1:
                features[name] = 0
                changed += 1
        if changed and not dry_run:
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return changed

    if update_build_config:
        build_path = get_feature_set_config_path(feature_set)
        c = update_one(build_path)
        total_changed += c
        if c:
            messages.append(f"build config ({c} disabled)")

    if update_train_config:
        train_path = get_train_features_config_path(feature_set)
        c = update_one(train_path)
        total_changed += c
        if c:
            messages.append(f"train config ({c} disabled)")

    if dry_run:
        return total_changed, f"Would disable {total_changed} feature(s) in configs."
    if total_changed == 0:
        return 0, "No enabled features in config matched the list (already disabled or not present)."
    return total_changed, f"Disabled {total_changed} feature(s) in: " + ", ".join(messages)
