#!/usr/bin/env python3
"""
analyze_features.py

Comprehensive feature analysis tool for identifying:
1. Feature importance distribution
2. Feature correlations and redundancy
3. Low-value features to prune
4. Redundant feature pairs

This script helps optimize the feature set by identifying:
- Features with very low importance (noise)
- Highly correlated features (redundancy)
- Features with high NaN rates
- Constant/near-constant features
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import joblib

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "features_labeled"
TRAIN_CFG = PROJECT_ROOT / "config" / "train_features.yaml"

# Importance thresholds
HIGH_IMPORTANCE_THRESHOLD = 0.01
MEDIUM_IMPORTANCE_THRESHOLD = 0.001
LOW_IMPORTANCE_THRESHOLD = 0.0001

# Correlation threshold for redundancy
CORRELATION_THRESHOLD = 0.95

# NaN threshold (features with >10% NaN are problematic)
NAN_THRESHOLD = 0.10


def load_training_metadata(model_path: Path) -> Dict:
    """Load training metadata from model file or JSON."""
    # Try loading from model file first
    if model_path.exists():
        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and "metadata" in model_data:
                return model_data["metadata"]
        except Exception as e:
            print(f"Warning: Could not load from model file: {e}")
    
    # Try loading from JSON metadata file
    metadata_file = MODEL_DIR / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Could not find training metadata in {model_path} or {metadata_file}")


def analyze_feature_importance(importances: Dict[str, float]) -> Dict:
    """
    Analyze feature importance distribution.
    
    Returns:
        Dictionary with categorized features and statistics
    """
    # Sort features by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    # Categorize features
    high_importance = []
    medium_importance = []
    low_importance = []
    very_low_importance = []
    zero_importance = []
    
    for feat, imp in sorted_features:
        if imp == 0.0:
            zero_importance.append((feat, imp))
        elif imp < LOW_IMPORTANCE_THRESHOLD:
            very_low_importance.append((feat, imp))
        elif imp < MEDIUM_IMPORTANCE_THRESHOLD:
            low_importance.append((feat, imp))
        elif imp < HIGH_IMPORTANCE_THRESHOLD:
            medium_importance.append((feat, imp))
        else:
            high_importance.append((feat, imp))
    
    # Calculate statistics
    total_features = len(sorted_features)
    total_importance = sum(imp for _, imp in sorted_features)
    
    high_importance_pct = sum(imp for _, imp in high_importance) / total_importance * 100 if total_importance > 0 else 0
    medium_importance_pct = sum(imp for _, imp in medium_importance) / total_importance * 100 if total_importance > 0 else 0
    low_importance_pct = sum(imp for _, imp in low_importance) / total_importance * 100 if total_importance > 0 else 0
    
    return {
        "sorted_features": sorted_features,
        "high_importance": high_importance,
        "medium_importance": medium_importance,
        "low_importance": low_importance,
        "very_low_importance": very_low_importance,
        "zero_importance": zero_importance,
        "statistics": {
            "total_features": total_features,
            "high_count": len(high_importance),
            "medium_count": len(medium_importance),
            "low_count": len(low_importance),
            "very_low_count": len(very_low_importance),
            "zero_count": len(zero_importance),
            "high_importance_pct": high_importance_pct,
            "medium_importance_pct": medium_importance_pct,
            "low_importance_pct": low_importance_pct,
            "top_20_cumulative_pct": sum(imp for _, imp in sorted_features[:20]) / total_importance * 100 if total_importance > 0 else 0
        }
    }


def load_feature_data(sample_size: int = 10000) -> pd.DataFrame:
    """
    Load a sample of feature data for correlation analysis.
    
    Args:
        sample_size: Number of rows to sample (for performance)
    
    Returns:
        DataFrame with features
    """
    print(f"Loading feature data (sampling {sample_size:,} rows)...")
    
    all_data = []
    count = 0
    
    for parquet_file in DATA_DIR.glob("*.parquet"):
        if count >= sample_size:
            break
        
        df = pd.read_parquet(parquet_file)
        # Sample rows if file is large
        if len(df) > sample_size // 10:  # If file has more than 1/10 of sample size
            df = df.sample(n=min(sample_size // 10, len(df)), random_state=42)
        
        all_data.append(df)
        count += len(df)
    
    if not all_data:
        raise FileNotFoundError(f"No feature files found in {DATA_DIR}")
    
    combined = pd.concat(all_data, ignore_index=False)
    
    # Sample if we have too much data
    if len(combined) > sample_size:
        combined = combined.sample(n=sample_size, random_state=42)
    
    print(f"Loaded {len(combined):,} rows with {len(combined.columns)} columns")
    return combined


def calculate_correlations(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for features.
    
    Args:
        df: DataFrame with feature data
        features: List of feature names to analyze
    
    Returns:
        Correlation matrix
    """
    print(f"Calculating correlations for {len(features)} features...")
    
    # Select only features that exist in the data
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: {len(missing)} features not found in data: {list(missing)[:5]}...")
    
    # Calculate correlation
    corr_matrix = df[available_features].corr()
    
    return corr_matrix


def find_redundant_pairs(corr_matrix: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> List[Tuple[str, str, float]]:
    """
    Find highly correlated feature pairs (redundant features).
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold (default: 0.95)
    
    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    redundant_pairs = []
    
    # Get upper triangle (avoid duplicates and diagonal)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            
            if abs(corr) >= threshold:
                redundant_pairs.append((feat1, feat2, corr))
    
    # Sort by absolute correlation (highest first)
    redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return redundant_pairs


def analyze_feature_quality(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Analyze feature quality (NaN rates, variance, etc.).
    
    Args:
        df: DataFrame with feature data
        features: List of feature names
    
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {}
    
    available_features = [f for f in features if f in df.columns]
    
    for feat in available_features:
        series = df[feat]
        
        # Calculate metrics
        nan_rate = series.isna().sum() / len(series)
        variance = series.var()
        std = series.std()
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val if not pd.isna(max_val) and not pd.isna(min_val) else np.nan
        
        # Check if constant or near-constant
        is_constant = variance < 1e-10 or range_val < 1e-10
        is_near_constant = variance < 1e-6 or range_val < 1e-6
        
        quality_metrics[feat] = {
            "nan_rate": nan_rate,
            "variance": variance,
            "std": std,
            "range": range_val,
            "is_constant": is_constant,
            "is_near_constant": is_near_constant,
            "has_high_nan": nan_rate > NAN_THRESHOLD
        }
    
    return quality_metrics


def generate_pruning_recommendations(
    importance_analysis: Dict,
    redundant_pairs: List[Tuple[str, str, float]],
    quality_metrics: Dict,
    feature_importances: Dict[str, float]
) -> Dict:
    """
    Generate pruning recommendations based on all analyses.
    
    Returns:
        Dictionary with pruning recommendations
    """
    recommendations = {
        "remove_zero_importance": [],
        "remove_very_low_importance": [],
        "remove_low_importance": [],
        "remove_redundant": [],
        "remove_high_nan": [],
        "remove_constant": [],
        "keep_redundant": []  # Features to keep from redundant pairs
    }
    
    # 1. Remove zero importance features
    for feat, imp in importance_analysis["zero_importance"]:
        recommendations["remove_zero_importance"].append(feat)
    
    # 2. Remove very low importance features
    for feat, imp in importance_analysis["very_low_importance"]:
        recommendations["remove_very_low_importance"].append(feat)
    
    # 3. Remove low importance features (optional - more aggressive)
    # We'll be conservative and only recommend very low importance
    
    # 4. Handle redundant pairs - keep the one with higher importance
    for feat1, feat2, corr in redundant_pairs:
        imp1 = feature_importances.get(feat1, 0.0)
        imp2 = feature_importances.get(feat2, 0.0)
        
        if imp1 > imp2:
            recommendations["remove_redundant"].append(feat2)
            recommendations["keep_redundant"].append((feat1, feat2, corr, imp1, imp2))
        else:
            recommendations["remove_redundant"].append(feat1)
            recommendations["keep_redundant"].append((feat2, feat1, corr, imp2, imp1))
    
    # 5. Remove features with high NaN rates
    for feat, metrics in quality_metrics.items():
        if metrics["has_high_nan"]:
            recommendations["remove_high_nan"].append(feat)
    
    # 6. Remove constant features
    for feat, metrics in quality_metrics.items():
        if metrics["is_constant"]:
            recommendations["remove_constant"].append(feat)
    
    # Combine all removal recommendations (avoid duplicates)
    all_to_remove = set()
    all_to_remove.update(recommendations["remove_zero_importance"])
    all_to_remove.update(recommendations["remove_very_low_importance"])
    all_to_remove.update(recommendations["remove_redundant"])
    all_to_remove.update(recommendations["remove_high_nan"])
    all_to_remove.update(recommendations["remove_constant"])
    
    recommendations["total_to_remove"] = len(all_to_remove)
    recommendations["features_to_remove"] = sorted(list(all_to_remove))
    
    return recommendations


def print_analysis_report(
    importance_analysis: Dict,
    redundant_pairs: List[Tuple[str, str, float]],
    quality_metrics: Dict,
    recommendations: Dict,
    total_features: int
):
    """Print comprehensive analysis report."""
    
    print("\n" + "="*80)
    print("FEATURE ANALYSIS REPORT")
    print("="*80)
    
    # Feature Importance Distribution
    print("\n### FEATURE IMPORTANCE DISTRIBUTION ###")
    stats = importance_analysis["statistics"]
    print(f"Total Features: {stats['total_features']}")
    print(f"\nImportance Categories:")
    print(f"  High importance (>={HIGH_IMPORTANCE_THRESHOLD}):     {stats['high_count']:3d} features ({stats['high_importance_pct']:.1f}% of total importance)")
    print(f"  Medium importance (>{MEDIUM_IMPORTANCE_THRESHOLD}):  {stats['medium_count']:3d} features ({stats['medium_importance_pct']:.1f}% of total importance)")
    print(f"  Low importance (>{LOW_IMPORTANCE_THRESHOLD}):       {stats['low_count']:3d} features ({stats['low_importance_pct']:.1f}% of total importance)")
    print(f"  Very low importance (<={LOW_IMPORTANCE_THRESHOLD}): {stats['very_low_count']:3d} features")
    print(f"  Zero importance:                                     {stats['zero_count']:3d} features")
    print(f"\nTop 20 features account for {stats['top_20_cumulative_pct']:.1f}% of total importance")
    
    # Top Features
    print(f"\n### TOP 20 FEATURES ###")
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Cumulative %':<12}")
    print("-" * 65)
    cumulative = 0
    total_imp = sum(imp for _, imp in importance_analysis["sorted_features"])
    for rank, (feat, imp) in enumerate(importance_analysis["sorted_features"][:20], 1):
        cumulative += imp
        pct = (imp / total_imp) * 100 if total_imp > 0 else 0
        cum_pct = (cumulative / total_imp) * 100 if total_imp > 0 else 0
        print(f"{rank:<6} {feat:<35} {imp:<12.6f} {cum_pct:<12.2f}")
    
    # Redundant Features
    print(f"\n### REDUNDANT FEATURE PAIRS (correlation >= {CORRELATION_THRESHOLD}) ###")
    if redundant_pairs:
        print(f"Found {len(redundant_pairs)} highly correlated pairs:")
        print(f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':<12}")
        print("-" * 72)
        for feat1, feat2, corr in redundant_pairs[:20]:  # Show top 20
            print(f"{feat1:<30} {feat2:<30} {corr:<12.4f}")
        if len(redundant_pairs) > 20:
            print(f"... and {len(redundant_pairs) - 20} more pairs")
    else:
        print("No highly correlated pairs found (good!)")
    
    # Quality Issues
    print(f"\n### FEATURE QUALITY ISSUES ###")
    high_nan = [f for f, m in quality_metrics.items() if m["has_high_nan"]]
    constant = [f for f, m in quality_metrics.items() if m["is_constant"]]
    near_constant = [f for f, m in quality_metrics.items() if m["is_near_constant"] and not m["is_constant"]]
    
    print(f"Features with high NaN rate (>{(NAN_THRESHOLD*100):.0f}%): {len(high_nan)}")
    if high_nan:
        print(f"  {', '.join(high_nan[:10])}{'...' if len(high_nan) > 10 else ''}")
    
    print(f"Constant features (no variance): {len(constant)}")
    if constant:
        print(f"  {', '.join(constant)}")
    
    print(f"Near-constant features: {len(near_constant)}")
    if near_constant:
        print(f"  {', '.join(near_constant[:10])}{'...' if len(near_constant) > 10 else ''}")
    
    # Pruning Recommendations
    print(f"\n### PRUNING RECOMMENDATIONS ###")
    rec = recommendations
    print(f"\nTotal features to remove: {rec['total_to_remove']}")
    print(f"  - Zero importance: {len(rec['remove_zero_importance'])}")
    print(f"  - Very low importance: {len(rec['remove_very_low_importance'])}")
    print(f"  - Redundant (lower importance): {len(rec['remove_redundant'])}")
    print(f"  - High NaN rate: {len(rec['remove_high_nan'])}")
    print(f"  - Constant: {len(rec['remove_constant'])}")
    
    print(f"\nRecommended feature count after pruning: {total_features - rec['total_to_remove']}")
    print(f"Reduction: {rec['total_to_remove']} features ({rec['total_to_remove']/total_features*100:.1f}%)")
    
    print(f"\n### FEATURES TO REMOVE ###")
    if rec["features_to_remove"]:
        # Group by reason
        print("\nBy Category:")
        if rec["remove_zero_importance"]:
            print(f"\nZero Importance ({len(rec['remove_zero_importance'])}):")
            for feat in rec["remove_zero_importance"][:10]:
                print(f"  - {feat}")
            if len(rec["remove_zero_importance"]) > 10:
                print(f"  ... and {len(rec['remove_zero_importance']) - 10} more")
        
        if rec["remove_very_low_importance"]:
            print(f"\nVery Low Importance ({len(rec['remove_very_low_importance'])}):")
            for feat in rec["remove_very_low_importance"][:10]:
                print(f"  - {feat}")
            if len(rec["remove_very_low_importance"]) > 10:
                print(f"  ... and {len(rec['remove_very_low_importance']) - 10} more")
        
        if rec["remove_redundant"]:
            print(f"\nRedundant (Lower Importance) ({len(rec['remove_redundant'])}):")
            for feat in rec["remove_redundant"][:10]:
                print(f"  - {feat}")
            if len(rec["remove_redundant"]) > 10:
                print(f"  ... and {len(rec['remove_redundant']) - 10} more")
        
        print(f"\nComplete List ({len(rec['features_to_remove'])} features):")
        for feat in rec["features_to_remove"]:
            print(f"  - {feat}")
    else:
        print("No features recommended for removal (all features appear valuable)")
    
    print("\n" + "="*80)


def save_recommendations(recommendations: Dict, output_file: Path):
    """Save pruning recommendations to JSON file."""
    # Convert to JSON-serializable format
    output = {
        "features_to_remove": recommendations["features_to_remove"],
        "total_to_remove": recommendations["total_to_remove"],
        "by_category": {
            "zero_importance": recommendations["remove_zero_importance"],
            "very_low_importance": recommendations["remove_very_low_importance"],
            "redundant": recommendations["remove_redundant"],
            "high_nan": recommendations["remove_high_nan"],
            "constant": recommendations["remove_constant"]
        },
        "redundant_pairs_kept": [
            {"keep": feat1, "remove": feat2, "correlation": corr, "keep_imp": imp1, "remove_imp": imp2}
            for feat1, feat2, corr, imp1, imp2 in recommendations["keep_redundant"]
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nRecommendations saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze features for pruning opportunities")
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_DIR / "xgb_classifier_selected_features.pkl"),
        help="Path to trained model file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of rows to sample for correlation analysis (default: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODEL_DIR / "feature_pruning_recommendations.json"),
        help="Output file for pruning recommendations (JSON)"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=CORRELATION_THRESHOLD,
        help=f"Correlation threshold for redundancy (default: {CORRELATION_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    print("="*80)
    print("FEATURE ANALYSIS")
    print("="*80)
    
    # Step 1: Load training metadata
    print("\n[1/5] Loading training metadata...")
    metadata = load_training_metadata(model_path)
    feature_importances = metadata.get("feature_importances", {})
    features = metadata.get("features", list(feature_importances.keys()))
    
    if not feature_importances:
        raise ValueError("No feature importances found in metadata")
    
    print(f"Loaded {len(features)} features from metadata")
    
    # Step 2: Analyze feature importance
    print("\n[2/5] Analyzing feature importance distribution...")
    importance_analysis = analyze_feature_importance(feature_importances)
    
    # Step 3: Load feature data and calculate correlations
    print("\n[3/5] Loading feature data and calculating correlations...")
    corr_matrix = None
    try:
        df = load_feature_data(args.sample_size)
        corr_matrix = calculate_correlations(df, features)
        redundant_pairs = find_redundant_pairs(corr_matrix, args.correlation_threshold)
        
        # Step 4: Analyze feature quality
        print("\n[4/5] Analyzing feature quality (NaN rates, variance)...")
        quality_metrics = analyze_feature_quality(df, features)
    except Exception as e:
        print(f"Warning: Could not load feature data for correlation analysis: {e}")
        print("Skipping correlation and quality analysis...")
        redundant_pairs = []
        quality_metrics = {}
    
    # Step 5: Generate recommendations
    print("\n[5/5] Generating pruning recommendations...")
    recommendations = generate_pruning_recommendations(
        importance_analysis,
        redundant_pairs,
        quality_metrics,
        feature_importances
    )
    
    # Print report
    print_analysis_report(
        importance_analysis,
        redundant_pairs,
        quality_metrics,
        recommendations,
        len(features)
    )
    
    # Save recommendations
    output_file = Path(args.output)
    save_recommendations(recommendations, output_file)
    
    # Export all feature importances to CSV
    fi_csv_file = MODEL_DIR / "feature_importances_all.csv"
    sorted_features = importance_analysis["sorted_features"]
    total_imp = sum(imp for _, imp in sorted_features)
    fi_df = pd.DataFrame([
        {
            'feature': feat,
            'importance': imp,
            'rank': rank + 1,
            'importance_pct': (imp / total_imp * 100) if total_imp > 0 else 0,
            'cumulative_pct': sum(imp2 for _, imp2 in sorted_features[:rank+1]) / total_imp * 100 if total_imp > 0 else 0,
            'category': (
                'high' if imp >= HIGH_IMPORTANCE_THRESHOLD else
                'medium' if imp >= MEDIUM_IMPORTANCE_THRESHOLD else
                'low' if imp >= LOW_IMPORTANCE_THRESHOLD else
                'very_low' if imp > 0 else 'zero'
            )
        }
        for rank, (feat, imp) in enumerate(sorted_features)
    ])
    fi_df['importance_pct'] = fi_df['importance_pct'].round(4)
    fi_df['cumulative_pct'] = fi_df['cumulative_pct'].round(2)
    fi_df.to_csv(fi_csv_file, index=False)
    print(f"\nAll {len(fi_df)} feature importances exported to: {fi_csv_file}")
    
    # Also export correlation matrix if available
    if corr_matrix is not None:
        corr_csv_file = MODEL_DIR / "feature_correlations.csv"
        corr_matrix.to_csv(corr_csv_file)
        print(f"Feature correlation matrix exported to: {corr_csv_file}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

