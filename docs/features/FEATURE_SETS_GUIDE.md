# Feature Sets Guide

## Overview

The feature set system allows you to create and manage multiple feature configurations for experimentation. This enables you to:
- Keep your current working feature set intact
- Build new feature sets with different configurations
- Compare performance across different feature sets
- Gradually refine and replace feature sets

## Quick Start

### 1. List Available Feature Sets

```bash
python src/manage_feature_sets.py list
```

This shows all available feature sets, including:
- Default set (v1) - your current feature set
- Any additional feature sets you've created

### 2. Create a New Feature Set

```bash
# Create a new feature set (copies from default v1)
python src/manage_feature_sets.py create v2 --description "Experimental feature set"

# Or copy from a specific feature set
python src/manage_feature_sets.py create v3 --copy-from v2 --description "Refined version"
```

This creates:
- `config/features_v2.yaml` - Feature configuration file
- `config/train_features_v2.yaml` - Training feature configuration (if it exists in source)
- `data/features_labeled_v2/` - Directory for feature data (created when you build features)

### 3. View Feature Set Information

```bash
python src/manage_feature_sets.py info v2
```

Shows detailed information about the feature set including:
- Config and data paths
- Number of enabled features
- Number of data files
- Metadata

### 4. Build Features for a Specific Feature Set

```bash
# Using the main app
python src/swing_trade_app.py features --feature-set v2 --horizon 30 --threshold 0.15

# Or directly with feature_pipeline.py
python src/feature_pipeline.py --feature-set v2 --horizon 30 --threshold 0.15
```

This will:
- Use `config/features_v2.yaml` for feature configuration
- Save features to `data/features_labeled_v2/`
- Keep your original feature set (v1) untouched

### 5. Edit Feature Set Configuration

Edit the feature set's config file to enable/disable features:

```bash
# Edit v2 feature set
notepad config/features_v2.yaml

# Or use any editor
code config/features_v2.yaml
```

Then rebuild features:
```bash
python src/swing_trade_app.py features --feature-set v2 --horizon 30 --threshold 0.15 --full
```

## Feature Set Structure

### Default Feature Set (v1)

The default feature set uses the original paths for backward compatibility:
- **Config**: `config/features.yaml`
- **Data**: `data/features_labeled/`
- **Train Config**: `config/train_features.yaml`

### Custom Feature Sets (v2, v3, etc.)

Custom feature sets use suffixed paths:
- **Config**: `config/features_<name>.yaml`
- **Data**: `data/features_labeled_<name>/`
- **Train Config**: `config/train_features_<name>.yaml`

## Workflow Example

### Scenario: Building a Better Feature Set

1. **Start with current set as baseline:**
   ```bash
   # Your current set is v1 (default)
   python src/manage_feature_sets.py list
   ```

2. **Create a new experimental set:**
   ```bash
   python src/manage_feature_sets.py create v2 --description "Experimental set with new features"
   ```

3. **Edit the new feature set:**
   ```bash
   # Open config/features_v2.yaml and enable/disable features
   # Remove redundant features, add new ones, etc.
   ```

4. **Build features for the new set:**
   ```bash
   python src/swing_trade_app.py features --feature-set v2 --horizon 30 --threshold 0.15
   ```

5. **Train model with new features:**
   ```bash
   # Update config/train_features_v2.yaml to select features
   # Then train (you'll need to update train_model.py to support feature sets)
   ```

6. **Compare performance:**
   - Run backtests with both feature sets
   - Compare metrics
   - Keep the better performing set

7. **When ready, replace the default:**
   ```bash
   # Option 1: Copy v2 configs to default locations
   copy config\features_v2.yaml config\features.yaml
   copy config\train_features_v2.yaml config\train_features.yaml
   
   # Option 2: Keep both and use --feature-set v2 going forward
   ```

## Command Reference

### Manage Feature Sets

```bash
# List all feature sets
python src/manage_feature_sets.py list

# Create new feature set
python src/manage_feature_sets.py create <name> [--copy-from <source>] [--description <text>]

# View feature set info
python src/manage_feature_sets.py info <name>
```

### Build Features

```bash
# Using feature set
python src/swing_trade_app.py features --feature-set v2 --horizon 30 --threshold 0.15

# Using explicit paths (backward compatible)
python src/swing_trade_app.py features --config config/features.yaml --output-dir data/features_labeled --horizon 30 --threshold 0.15
```

### Direct Pipeline Access

```bash
# Using feature set
python src/feature_pipeline.py --feature-set v2 --horizon 30 --threshold 0.15

# Using explicit paths
python src/feature_pipeline.py --input-dir data/clean --output-dir data/features_labeled --config config/features.yaml --horizon 30 --threshold 0.15
```

## Best Practices

1. **Keep the default set (v1) as your production baseline**
   - Don't modify it until you're confident in a replacement

2. **Use descriptive names for experimental sets**
   - `v2`, `v3` for sequential versions
   - `experimental`, `minimal`, `extended` for different approaches
   - `test_<date>` for temporary experiments

3. **Document your changes**
   - Use `--description` when creating feature sets
   - Add comments in config files explaining why features were enabled/disabled

4. **Test incrementally**
   - Make small changes between feature sets
   - Compare performance at each step
   - Keep notes on what works and what doesn't

5. **Clean up old feature sets**
   - Remove feature sets that didn't improve performance
   - Keep only the best performing sets for reference

## Integration with Other Scripts

Currently, feature sets are supported in:
- ✅ `feature_pipeline.py` - Feature building
- ✅ `swing_trade_app.py` - Main app features command
- ✅ `manage_feature_sets.py` - Feature set management

**Coming soon:**
- Training scripts (will need `--feature-set` parameter)
- Backtest scripts (will need `--feature-set` parameter)
- Identify scripts (will need `--feature-set` parameter)

## Troubleshooting

### Feature set not found
- Make sure you've created the feature set first: `python src/manage_feature_sets.py create <name>`
- Check that the config file exists: `config/features_<name>.yaml`

### Data directory not found
- This is normal - the data directory is created when you build features
- Run: `python src/swing_trade_app.py features --feature-set <name> ...`

### Import errors
- Make sure you're running from the project root
- Check that `src/feature_set_manager.py` exists

## Example: Complete Workflow

```bash
# 1. List current feature sets
python src/manage_feature_sets.py list

# 2. Create new experimental feature set
python src/manage_feature_sets.py create v2 --description "Minimal feature set - testing core indicators only"

# 3. Edit the config to remove features you don't want
# Edit config/features_v2.yaml and set unwanted features to 0

# 4. Build features for the new set
python src/swing_trade_app.py features --feature-set v2 --horizon 30 --threshold 0.15

# 5. Check the results
python src/manage_feature_sets.py info v2

# 6. When ready, train and backtest with the new set
# (Update train_model.py and enhanced_backtest.py to support --feature-set)
```

