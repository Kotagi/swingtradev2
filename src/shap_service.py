"""
SHAP Service - Model Explainability

Computes and manages SHAP (SHapley Additive exPlanations) values for model interpretability.
Provides lightweight artifact storage and retrieval for UI display.

Key features:
- TreeSHAP for fast computation (XGBoost-compatible)
- Uses validation set (out-of-sample) for realistic explanations
- Stratified sampling for balanced representation
- Lightweight artifact storage (summaries, not full matrices)
- Feature matching validation to ensure consistency
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SHAPService:
    """Service for computing and managing SHAP explanations."""
    
    # Default configuration
    DEFAULT_SAMPLE_SIZE = 1000  # Max samples for SHAP computation
    DEFAULT_MAX_SAMPLES = 2000  # Hard limit to prevent excessive computation
    
    def __init__(self, artifacts_dir: Optional[Path] = None):
        """
        Initialize SHAP service.
        
        Args:
            artifacts_dir: Directory for SHAP artifacts. Defaults to models/shap_artifacts/
        """
        if artifacts_dir is None:
            # Default to models/shap_artifacts/ relative to project root
            project_root = Path(__file__).parent.parent
            artifacts_dir = project_root / "models" / "shap_artifacts"
        
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_shap(
        self,
        model: Any,  # XGBClassifier
        X_data: pd.DataFrame,
        y_data: pd.Series,
        features: List[str],
        model_id: str,
        sample_size: Optional[int] = None,
        use_stratified: bool = True,
        data_split: str = "validation"
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a trained model.
        
        Args:
            model: Trained XGBoost model
            X_data: Feature data (validation set recommended)
            y_data: Target labels (for stratified sampling)
            features: List of feature names (must match X_data columns)
            model_id: Unique identifier for this model (used for artifact storage)
            sample_size: Number of samples to use (default: DEFAULT_SAMPLE_SIZE)
            use_stratified: Whether to use stratified sampling (default: True)
            data_split: Name of data split used (e.g., "validation", "test")
        
        Returns:
            Dictionary containing:
            - success: bool
            - message: str
            - artifacts_path: Path to saved artifacts (if successful)
            - metadata: Dict with computation details
        """
        if not HAS_SHAP:
            return {
                "success": False,
                "message": "SHAP library not installed. Install with: pip install shap",
                "artifacts_path": None,
                "metadata": None
            }
        
        if sample_size is None:
            sample_size = self.DEFAULT_SAMPLE_SIZE
        
        # Limit sample size to prevent excessive computation
        sample_size = min(sample_size, self.DEFAULT_MAX_SAMPLES, len(X_data))
        
        try:
            # Validate feature matching
            if set(features) != set(X_data.columns):
                missing = set(features) - set(X_data.columns)
                extra = set(X_data.columns) - set(features)
                return {
                    "success": False,
                    "message": f"Feature mismatch. Missing: {missing}, Extra: {extra}",
                    "artifacts_path": None,
                    "metadata": None
                }
            
            # Ensure feature order matches
            X_data = X_data[features].copy()
            
            # Sample data for SHAP computation
            if len(X_data) > sample_size:
                if use_stratified and y_data is not None:
                    # Stratified sampling to maintain class balance
                    from sklearn.model_selection import train_test_split
                    X_sample, _, y_sample, _ = train_test_split(
                        X_data, y_data,
                        test_size=1 - (sample_size / len(X_data)),
                        stratify=y_data,
                        random_state=42
                    )
                else:
                    # Simple random sampling
                    X_sample = X_data.sample(n=sample_size, random_state=42)
                    y_sample = y_data.loc[X_sample.index] if y_data is not None else None
            else:
                X_sample = X_data
                y_sample = y_data
            
            print(f"Computing SHAP values for {len(X_sample)} samples...")
            
            # Use TreeExplainer for fast computation (XGBoost-compatible)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classification (SHAP returns list for binary)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class SHAP values
            
            # Compute global importance (mean absolute SHAP per feature)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance ranking
            importance_ranking = sorted(
                zip(features, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Convert to percentage of total importance
            total_importance = sum(mean_abs_shap)
            importance_with_pct = [
                {
                    "feature": feat,
                    "importance": float(imp),
                    "importance_pct": float(imp / total_importance * 100),
                    "rank": rank + 1
                }
                for rank, (feat, imp) in enumerate(importance_ranking)
            ]
            
            # Compute summary statistics
            top_n = 10
            top_features_importance = sum([item["importance_pct"] for item in importance_with_pct[:top_n]])
            
            # Create metadata
            metadata = {
                "model_id": model_id,
                "computation_date": datetime.now().isoformat(),
                "data_split": data_split,
                "sample_size": len(X_sample),
                "total_samples_available": len(X_data),
                "n_features": len(features),
                "feature_list_hash": self._hash_feature_list(features),
                "top_n_concentration": {
                    "n": top_n,
                    "percentage": float(top_features_importance)
                },
                "shap_version": shap.__version__ if hasattr(shap, '__version__') else "unknown"
            }
            
            # Create summary data
            summary = {
                "importance_ranking": importance_with_pct,
                "global_mean_abs_shap": {feat: float(imp) for feat, imp in zip(features, mean_abs_shap)},
                "statistics": {
                    "total_importance": float(total_importance),
                    "mean_importance": float(np.mean(mean_abs_shap)),
                    "std_importance": float(np.std(mean_abs_shap)),
                    "max_importance": float(np.max(mean_abs_shap)),
                    "min_importance": float(np.min(mean_abs_shap))
                }
            }
            
            # Save artifacts
            artifacts_path = self._save_artifacts(
                model_id=model_id,
                summary=summary,
                metadata=metadata,
                shap_values=shap_values,
                X_sample=X_sample,
                features=features
            )
            
            return {
                "success": True,
                "message": f"SHAP computation complete. Artifacts saved to {artifacts_path}",
                "artifacts_path": artifacts_path,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error computing SHAP: {str(e)}",
                "artifacts_path": None,
                "metadata": None
            }
    
    def _save_artifacts(
        self,
        model_id: str,
        summary: Dict,
        metadata: Dict,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame,
        features: List[str]
    ) -> Path:
        """
        Save SHAP artifacts to disk.
        
        Args:
            model_id: Unique model identifier
            summary: Summary data (importance ranking, statistics)
            metadata: Computation metadata
            shap_values: Computed SHAP values (n_samples x n_features)
            X_sample: Sampled feature data used for computation
            features: Feature names
        
        Returns:
            Path to artifacts directory
        """
        # Create model-specific directory
        model_artifacts_dir = self.artifacts_dir / model_id
        model_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary JSON (lightweight, always saved)
        summary_file = model_artifacts_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save metadata JSON
        metadata_file = model_artifacts_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save importance ranking (for quick access)
        ranking_file = model_artifacts_dir / "importance_ranking.json"
        with open(ranking_file, 'w') as f:
            json.dump(summary["importance_ranking"], f, indent=2)
        
        # Generate and save summary plot (beeswarm-style)
        if HAS_MATPLOTLIB:
            try:
                plot_file = model_artifacts_dir / "summary_plot.png"
                self._create_summary_plot(shap_values, X_sample, features, plot_file)
            except Exception as e:
                print(f"Warning: Could not create SHAP plot: {e}")
        
        # Optionally save full SHAP values (only if small enough)
        # For now, we skip this to keep artifacts lightweight
        # Users can recompute if they need detailed per-sample SHAP
        
        return model_artifacts_dir
    
    def _create_summary_plot(
        self,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame,
        features: List[str],
        output_file: Path
    ):
        """
        Create a SHAP summary plot (beeswarm-style).
        
        Args:
            shap_values: SHAP values array (n_samples x n_features)
            X_sample: Feature data
            features: Feature names
            output_file: Path to save plot
        """
        # Create SHAP Explanation object for plotting
        shap_explanation = shap.Explanation(
            values=shap_values,
            base_values=np.zeros(len(X_sample)),  # Base value for binary classification
            data=X_sample.values,
            feature_names=features
        )
        
        # Create plot (show top 20 features)
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def load_artifacts(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load SHAP artifacts for a model.
        
        Args:
            model_id: Unique model identifier
        
        Returns:
            Dictionary with artifacts, or None if not found
        """
        model_artifacts_dir = self.artifacts_dir / model_id
        
        if not model_artifacts_dir.exists():
            return None
        
        try:
            artifacts = {}
            
            # Load summary
            summary_file = model_artifacts_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    artifacts["summary"] = json.load(f)
            
            # Load metadata
            metadata_file = model_artifacts_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    artifacts["metadata"] = json.load(f)
            
            # Load importance ranking
            ranking_file = model_artifacts_dir / "importance_ranking.json"
            if ranking_file.exists():
                with open(ranking_file, 'r') as f:
                    artifacts["importance_ranking"] = json.load(f)
            
            # Check for plot
            plot_file = model_artifacts_dir / "summary_plot.png"
            if plot_file.exists():
                artifacts["plot_path"] = str(plot_file)
            
            return artifacts if artifacts else None
            
        except Exception as e:
            print(f"Error loading SHAP artifacts: {e}")
            return None
    
    def artifact_exists(self, model_id: str) -> bool:
        """Check if SHAP artifacts exist for a model."""
        model_artifacts_dir = self.artifacts_dir / model_id
        return model_artifacts_dir.exists() and (model_artifacts_dir / "summary.json").exists()
    
    def delete_artifacts(self, model_id: str) -> bool:
        """Delete SHAP artifacts for a model."""
        model_artifacts_dir = self.artifacts_dir / model_id
        if model_artifacts_dir.exists():
            import shutil
            shutil.rmtree(model_artifacts_dir)
            return True
        return False
    
    def _hash_feature_list(self, features: List[str]) -> str:
        """Create a hash of the feature list for validation."""
        features_str = ",".join(sorted(features))
        return hashlib.md5(features_str.encode()).hexdigest()
    
    def validate_feature_match(self, model_features: List[str], shap_metadata: Dict) -> Tuple[bool, str]:
        """
        Validate that model features match SHAP computation features.
        
        Args:
            model_features: Current model feature list
            shap_metadata: SHAP metadata dict (from load_artifacts)
        
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        if shap_metadata is None:
            return False, "No SHAP metadata available"
        
        current_hash = self._hash_feature_list(model_features)
        stored_hash = shap_metadata.get("feature_list_hash")
        
        if stored_hash is None:
            return True, "No feature hash stored (old SHAP artifact)"
        
        if current_hash != stored_hash:
            return False, "Feature list mismatch - SHAP artifacts may be stale"
        
        return True, "Feature list matches"

